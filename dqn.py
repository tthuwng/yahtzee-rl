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

class PrioritizedNstepReplayBuffer:
    """
    A prioritized N-step replay buffer storing transitions with priority
    for improved sampling of rare or high-error experiences.
    """

    def __init__(self, capacity=300000, n_step=5, gamma=0.99, alpha=0.6, beta=0.4):
        """
        :param capacity: Max number of transitions to store
        :param n_step: Number of steps for multi-step returns
        :param gamma: Discount factor
        :param alpha: Priority exponent
        :param beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.nstep_queue = deque(maxlen=n_step)
        self.max_priority = 1.0

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
        # If we have n-step transitions, save them
        if len(self.nstep_queue) == self.n_step or done:
            n_s, n_a, n_r, n_next, n_done = self._calc_nstep_return(
                list(self.nstep_queue)
            )
            if len(self.buffer) < self.capacity:
                self.buffer.append(Transition(n_s, n_a, n_r, n_next, n_done))
                self.priorities[len(self.buffer) - 1] = self.max_priority
            else:
                self.buffer[self.pos] = Transition(n_s, n_a, n_r, n_next, n_done)
                self.priorities[self.pos] = self.max_priority
                self.pos = (self.pos + 1) % self.capacity

            if done:
                self.nstep_queue.clear()

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return None

        # Compute probabilities from priorities
        priorities = self.priorities[: len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        transitions = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)

        # Compute importance sampling weights
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states = np.stack([tr.state for tr in transitions])
        actions = np.array([tr.action for tr in transitions])
        rewards = np.array([tr.reward for tr in transitions], dtype=np.float32)
        next_states = np.stack([tr.next_state for tr in transitions])
        dones = np.array([tr.done for tr in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled transitions based on new TD errors.
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(prio, 1e-6)
            self.max_priority = max(self.max_priority, prio)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    Dueling DQN with three linear layers + dueling heads for value & advantage.
    """

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Feature layers
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)

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
        x = F.relu(self.fc3(x))

        val = self.value_stream(x)  # [B, 1]
        adv = self.adv_stream(x)    # [B, action_size]
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


class YahtzeeAgent:
    """N-step DQN agent with simpler net architecture + Prioritized Replay."""

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
        alpha: float = 0.6,   # PER alpha
        beta: float = 0.4,    # PER beta
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
        # Use prioritized replay
        self.buffer = PrioritizedNstepReplayBuffer(
            capacity=300000, n_step=n_step, gamma=gamma,
            alpha=alpha, beta=beta
        )

        # For multi-environment training if needed
        self.num_envs = num_envs

        self.epsilon = 1.0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.learn_steps = 0
        self.target_update = target_update
        self.training_mode = True
        self.alpha = alpha
        self.beta = beta

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
        dones: List[bool]
    ) -> float:
        """
        Train on a batch of transitions with encoded states + PER updates.
        """
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            s = np.asarray(s, dtype=np.float32)
            ns = np.asarray(ns, dtype=np.float32)
            self.buffer.push(s, a, r, ns, d)

        if len(self.buffer) < self.batch_size:
            return 0.0

        sample_result = self.buffer.sample(self.batch_size)
        if sample_result is None:
            return 0.0

        states_np, actions_np, rewards_np, next_states_np, dones_np, indices, weights = sample_result

        # Convert to tensors
        states_t = torch.from_numpy(states_np).float().to(self.device)
        actions_t = torch.from_numpy(actions_np).long().to(self.device)
        rewards_t = torch.from_numpy(rewards_np).float().to(self.device)
        next_states_t = torch.from_numpy(next_states_np).float().to(self.device)
        dones_t = torch.from_numpy(dones_np).float().to(self.device)
        weights_t = torch.from_numpy(weights).float().to(self.device)

        # Current Q
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Next Q (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + (1 - dones_t) * (self.gamma ** self.n_step) * next_q

        # Compute TD error for PER
        td_error = target_q - current_q
        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        new_priorities = td_error.detach().abs().cpu().numpy()
        self.buffer.update_priorities(indices, new_priorities)

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
            [state], [action], [reward], [next_state], [done]
        )

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "beta": self.beta,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"], strict=False)
        self.target_net.load_state_dict(ckpt["target_net"], strict=False)
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except:
                pass
        self.epsilon = ckpt.get("epsilon", 1.0)
        self.alpha = ckpt.get("alpha", 0.6)
        self.beta = ckpt.get("beta", 0.4)