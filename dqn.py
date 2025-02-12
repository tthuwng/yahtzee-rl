import random
from collections import deque, namedtuple
from typing import List


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class NstepReplayBuffer:
    """
    Simple N-step replay buffer (no priority for simplicity).
    Just store n-step transitions and pop them.
    """

    def __init__(self, capacity: int = 300000, n_step: int = 5, gamma: float = 0.99):
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
        for tr in transitions:
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
            n_s, n_a, n_r, n_next, n_done = self._calc_nstep_return(
                list(self.nstep_queue)
            )
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
    """Simpler DQN with two linear layers + dueling heads."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Feature layers
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.state_size:
            raise ValueError(
                f"Wrong input size, expected {self.state_size} got {x.shape[-1]}"
            )

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        val = self.value_stream(x)  # [B, 1]
        adv = self.adv_stream(x)  # [B, action_size]
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


class YahtzeeAgent:
    """N-step DQN agent with simpler net architecture."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 1024,
        gamma: float = 0.99,
        lr: float = 3e-4,
        device: str = "cuda",
        n_step: int = 5,
        target_update: int = 500,
        min_epsilon: float = 0.005,
        epsilon_decay: float = 0.9997,
        num_envs: int = 32,
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
        self.buffer = NstepReplayBuffer(capacity=300000, n_step=n_step, gamma=gamma)

        # Initialize n-step buffers for each environment
        self.num_envs = num_envs
        self.nstep_buffers = [deque(maxlen=n_step) for _ in range(num_envs)]

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
        env_indices: List[int],
    ) -> float:
        """Train on a batch of transitions."""
        # Store transitions in buffer
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.buffer.push(s, a, r, ns, d)

        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample and train
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return 0.0

        states_np, actions_np, rewards_np, next_states_np, dones_np = batch

        # Convert to tensors
        states_t = torch.from_numpy(states_np).float().to(self.device)
        actions_t = torch.from_numpy(actions_np).long().to(self.device)
        rewards_t = torch.from_numpy(rewards_np).float().to(self.device)
        next_states_t = torch.from_numpy(next_states_np).float().to(self.device)
        dones_t = torch.from_numpy(dones_np).float().to(self.device)

        # Current Q
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))

        # Next Q (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = (
                rewards_t.unsqueeze(1)
                + (1 - dones_t.unsqueeze(1)) * ((self.gamma) ** self.n_step) * next_q
            )

        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return loss.item()

    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            st = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
            q = self.policy_net(st).squeeze(0).cpu().numpy()
        return q

    def train_step(self, state, action, reward, next_state, done) -> float:
        return self.train_step_batch(
            [state], [action], [reward], [next_state], [done], [0]
        )

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

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
        self.epsilon = ckpt.get("epsilon", 1.0)
