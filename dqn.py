import random
from collections import deque, namedtuple
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class NstepReplayBuffer:
    """
    Simple N-step replay buffer (no priority for simplicity).
    Just store n-step transitions and pop them.
    """
    def __init__(self, capacity: int = 100000, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        self.buffer = []
        self.pos = 0

        self.nstep_queue = deque(maxlen=n_step)

    def _calc_nstep_return(self, transitions: List[Transition]):
        """Compute discounted n-step return."""
        R = 0.0
        discount = 1.0
        done_final = False
        s0, a0 = transitions[0].state, transitions[0].action
        next_s_final = transitions[-1].next_state
        for i, tr in enumerate(transitions):
            R += discount * tr.reward
            discount *= self.gamma
            if tr.done:
                done_final = True
                next_s_final = tr.next_state
                break
        return s0, a0, R, next_s_final, done_final

    def push(self, state, action, reward, next_state, done):
        self.nstep_queue.append(Transition(state, action, reward, next_state, done))
        # if we have n-step transitions, save them
        if len(self.nstep_queue) == self.n_step or done:
            # compute n-step return
            n_s, n_a, n_r, n_next, n_done = self._calc_nstep_return(list(self.nstep_queue))
            # store
            if len(self.buffer) < self.capacity:
                self.buffer.append(Transition(n_s, n_a, n_r, n_next, n_done))
            else:
                self.buffer[self.pos] = Transition(n_s, n_a, n_r, n_next, n_done)
                self.pos = (self.pos + 1) % self.capacity

            if done:
                self.nstep_queue.clear()

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return None
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[idx] for idx in indices]

        states = np.stack([b.state for b in batch])
        actions = np.array([b.action for b in batch])
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.stack([b.next_state for b in batch])
        dones = np.array([b.done for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """ Wider net, no dropout, residual blocks. """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Input
        self.input_layer = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

        # 2 Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.LayerNorm(512)
            )
            self.res_blocks.append(block)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.state_size:
            raise ValueError(f"Wrong input size, expected {self.state_size} got {x.shape[-1]}")

        x = self.input_layer(x)
        for block in self.res_blocks:
            residual = x
            x = block(x) + residual

        val = self.value_stream(x)           # [B, 1]
        adv = self.adv_stream(x)             # [B, action_size]
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

class YahtzeeAgent:
    """N-step DQN agent with bigger net, ignoring mismatch keys on load."""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 512,
        gamma: float = 0.99,
        lr: float = 1e-4,
        device: str = "cuda",
        n_step: int = 3,
        target_update: int = 100,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.9995
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.n_step = n_step
        self.buffer = NstepReplayBuffer(capacity=200000, n_step=n_step, gamma=gamma)

        self.epsilon = 1.0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.learn_steps = 0
        self.target_update = target_update
        self.training_mode = True

    def train(self):
        self.training_mode = True
        self.policy_net.train()
        self.target_net.train()

    def eval(self):
        self.training_mode = False
        self.policy_net.eval()
        self.target_net.eval()

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, valid_actions: List[int]) -> int:
        """Basic epsilon-greedy with valid action masking."""
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        q_values = self.policy_net(state_t).squeeze(0)
        mask = torch.full((self.action_size,), float("-inf"), device=self.device)
        mask[valid_actions] = 0
        q_values = q_values + mask
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ) -> float:
        # store transitions
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.store_transition(s, a, r, ns, d)

        if len(self.buffer) < self.batch_size:
            return 0.0

        sample = self.buffer.sample(self.batch_size)
        if sample is None:
            return 0.0

        states_np, actions_np, rewards_np, next_states_np, dones_np = sample

        # convert to Tensors
        st = torch.from_numpy(states_np).float().to(self.device)
        ac = torch.from_numpy(actions_np).long().to(self.device)
        rw = torch.from_numpy(rewards_np).float().to(self.device)
        ns = torch.from_numpy(next_states_np).float().to(self.device)
        dn = torch.from_numpy(dones_np).float().to(self.device)

        # Q-learning with Double DQN
        q_values = self.policy_net(st).gather(1, ac.unsqueeze(1))  # shape [B,1]
        with torch.no_grad():
            next_actions = self.policy_net(ns).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions)
            target = rw.unsqueeze(1) + (1 - dn.unsqueeze(1)) * self.gamma * next_q

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return loss.item()

    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            st = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
            q = self.policy_net(st).squeeze(0).cpu().numpy()
        return q

    def train_step(self, state, action, reward, next_state, done) -> float:
        return self.train_step_batch([state],[action],[reward],[next_state],[done])

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        policy_dict = ckpt["policy_net"]
        target_dict = ckpt["target_net"]
        self.policy_net.load_state_dict(policy_dict, strict=False)
        self.target_net.load_state_dict(target_dict, strict=False)
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except:
                pass
        self.epsilon = ckpt.get("epsilon", 1.0)