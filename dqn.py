import random
from collections import namedtuple, deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "td_error")
)

class NoisyLinear(nn.Module):
    """
    Noisy linear layer for NoisyNet exploration (factorized Gaussian noise).
    Reference: https://arxiv.org/abs/1706.10295
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, use_cuda=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.sigma_init = sigma_init
        self.use_cuda = use_cuda
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))

    def reset_noise(self):
        eps_in = torch.randn(self.in_features)
        eps_out = torch.randn(self.out_features)

        # Factorized noise
        eps_in = eps_in.sign().mul_(eps_in.abs().sqrt_())
        eps_out = eps_out.sign().mul_(eps_out.abs().sqrt_())

        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DQN(nn.Module):
    """Dueling DQN with NoisyLinear for exploration."""

    def __init__(self, state_size: int, action_size: int, use_cuda=True) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Input layer
        self.input_layer = nn.Sequential(
            NoisyLinear(state_size, 512, use_cuda=use_cuda),
            nn.ReLU()
        )

        # Residual blocks replaced with a simpler approach + Noisy
        self.hidden_layer = nn.Sequential(
            NoisyLinear(512, 512, use_cuda=use_cuda),
            nn.ReLU(),
            NoisyLinear(512, 512, use_cuda=use_cuda),
            nn.ReLU(),
        )

        # Dueling streams: Noisy
        self.value_stream = nn.Sequential(
            NoisyLinear(512, 256, use_cuda=use_cuda),
            nn.ReLU(),
            NoisyLinear(256, 1, use_cuda=use_cuda),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 256, use_cuda=use_cuda),
            nn.ReLU(),
            NoisyLinear(256, action_size, use_cuda=use_cuda),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # shape check
        if state.shape[-1] != self.state_size:
            raise ValueError(f"Expected input of size {self.state_size}, got {state.shape[-1]}")

        x = self.input_layer(state)
        x_res = self.hidden_layer(x)
        x = x + x_res  # skip connection
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        # Reset noise in all NoisyLinear layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer. Storing single-step transitions, updated with n-step in agent."""

    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.7,
        beta: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.00001
        self.buffer = []
        self.device = torch.device(device)
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.pos = 0
        self.eps = 1e-5

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = 1.0,
    ) -> None:
        priority = (abs(td_error) + self.eps) ** self.alpha
        transition = Transition(state, action, reward, next_state, done, td_error)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities[len(self.buffer) - 1] = priority
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        n_samples = len(self.buffer)
        if n_samples == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        probs = self.priorities[:n_samples] / self.priorities[:n_samples].sum()
        indices = torch.multinomial(probs, batch_size, replacement=True)

        weights = (n_samples * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[idx.item()] for idx in indices]

        states = torch.from_numpy(np.stack([t.state for t in batch])).float().to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float).to(device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).float().to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float).to(device)
        weights = weights.to(device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: torch.Tensor, new_errors: torch.Tensor) -> None:
        indices = indices.to(self.device)
        new_errors = new_errors.to(self.device)
        priorities = (new_errors.abs() + self.eps) ** self.alpha
        self.priorities[indices] = priorities

    def __len__(self) -> int:
        return len(self.buffer)


class NstepBuffer:
    """
    Temporary buffer to accumulate transitions for n-step returns.
    We maintain one of these per environment in the agent.
    """
    def __init__(self, n=5, gamma=0.99):
        self.n = n
        self.gamma = gamma
        self.queue = []

    def append(self, s, a, r):
        self.queue.append((s, a, r))

    def pop(self):
        return self.queue.pop(0) if self.queue else None

    def is_full(self):
        return len(self.queue) >= self.n

    def get_nstep_transition(self, next_state, done):
        """
        Return a single n-step transition:
            (s0, a0, R, s_n, done_n)
        R = r0 + gamma*r1 + ... + gamma^(n-1)*r_(n-1)
        done_n is true if any of the steps ended in done.
        """
        total_reward = 0.0
        for i, (_, _, r) in enumerate(self.queue):
            total_reward += (self.gamma**i) * r

        s0, a0, _ = self.queue[0]
        return s0, a0, total_reward, next_state, done


class YahtzeeAgent:
    """Rainbow-ish DQN agent with n-step returns, NoisyNet, and PER."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 4096,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        target_update: int = 100,
        device: str = "cuda",
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.9995,
        n_step: int = 5,
        num_envs: int = 64
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.device = torch.device(device)
        self.training_mode = True

        # Epsilon is mostly overshadowed by NoisyNets, but we keep a small usage
        self.epsilon = 1.0
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Networks
        self.policy_net = DQN(state_size, action_size, use_cuda=(device=="cuda")).to(self.device)
        self.target_net = DQN(state_size, action_size, use_cuda=(device=="cuda")).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=1_000_000,
            alpha=0.7,
            beta=0.5,
            device=device
        )

        # Simple Adam (no scheduler)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # N-step for each environment
        self.n_step = n_step
        self.nstep_buffers = [NstepBuffer(n=n_step, gamma=gamma) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.learn_steps = 0

    def train(self):
        self.training_mode = True
        self.policy_net.train()

    def eval(self):
        self.training_mode = False
        self.policy_net.eval()

    @torch.no_grad()
    def select_actions_batch(
        self, state_vecs: np.ndarray, valid_actions_list: List[List[int]]
    ) -> List[int]:
        """
        Uses NoisyNet for exploration primarily.
        We also keep a small epsilon-greedy just in case.
        """
        batch_size = len(state_vecs)
        actions = [None] * batch_size

        # Epsilon exploration
        random_mask = np.random.rand(batch_size) < self.epsilon
        states_t = torch.from_numpy(state_vecs).float().to(self.device)
        q_values = self.policy_net(states_t)

        for i in range(batch_size):
            if random_mask[i] and self.training_mode:
                actions[i] = random.choice(valid_actions_list[i])
            else:
                # mask invalid
                mask = torch.full((self.action_size,), float('-inf'), device=self.device)
                mask[valid_actions_list[i]] = 0
                q_vals_masked = q_values[i] + mask
                actions[i] = q_vals_masked.argmax().item()

        return actions

    def _store_nstep_transition(self, env_idx, next_state, done):
        """
        Once n-step buffer for env_idx is full or done,
        generate an n-step transition and push to PER.
        """
        s0, a0, R, s_next, d_next = self.nstep_buffers[env_idx].get_nstep_transition(next_state, done)
        # push
        self.buffer.push(s0, a0, R, s_next, d_next, td_error=abs(R))

    def train_step_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
        env_indices: List[int],
    ) -> float:
        """
        We incorporate n-step transitions: each environment has its own NstepBuffer.
        """
        # Fill each environment's n-step buffer
        for idx, s, a, r in zip(env_indices, states, actions, rewards):
            self.nstep_buffers[idx].append(s, a, r)

        # For each environment, if done or buffer is full, produce an n-step transition
        for i, done_flag, ns in zip(env_indices, dones, next_states):
            if done_flag or self.nstep_buffers[i].is_full():
                self._store_nstep_transition(i, ns, done_flag)
                self.nstep_buffers[i].pop()

            # If done, flush all leftover transitions
            if done_flag:
                while self.nstep_buffers[i].queue:
                    self._store_nstep_transition(i, ns, True)
                    self.nstep_buffers[i].pop()

        # If buffer not big enough
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample from PER
        states_t, actions_t, rewards_t, next_states_t, dones_t, weights, indices = self.buffer.sample(self.batch_size, self.device)

        # train
        self.policy_net.reset_noise()  # reset noisy layers
        self.target_net.reset_noise()

        with torch.cuda.amp.autocast():
            current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                target_q_next = self.target_net(next_states_t).gather(1, next_actions)
                target_q = rewards_t.unsqueeze(1) + (1.0 - dones_t.unsqueeze(1)) * (self.gamma ** self.n_step) * target_q_next

            td_errors = (target_q - current_q).abs()
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.detach().squeeze())

        # Soft update every step
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(0.005 * policy_param.data + 0.995 * target_param.data)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    @torch.no_grad()
    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        st = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        q = self.policy_net(st).squeeze(0)
        return q.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, valid_actions: List[int]) -> int:
        """
        Single-env version of selecting action with NoisyNet + small epsilon.
        """
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)

        s_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        q_values = self.policy_net(s_t).squeeze(0)
        mask = torch.full((self.action_size,), float('-inf'), device=self.device)
        mask[valid_actions] = 0
        q_values = q_values + mask
        return q_values.argmax().item()

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        env_idx: int
    ) -> float:
        """
        Single environment version that also uses the n-step logic.
        Not used much if we do batch stepping, but included for completeness.
        """
        self.nstep_buffers[env_idx].append(state, action, reward)
        if done or self.nstep_buffers[env_idx].is_full():
            self._store_nstep_transition(env_idx, next_state, done)
            self.nstep_buffers[env_idx].pop()

        if done:
            while self.nstep_buffers[env_idx].queue:
                self._store_nstep_transition(env_idx, next_state, True)
                self.nstep_buffers[env_idx].pop()

        if len(self.buffer) < self.batch_size:
            return 0.0

        return self._train_on_batch()

    def _train_on_batch(self) -> float:
        # exactly as in train_step_batch but for single transitions
        states_t, actions_t, rewards_t, next_states_t, dones_t, weights, indices = self.buffer.sample(self.batch_size, self.device)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        with torch.cuda.amp.autocast():
            current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                target_q_next = self.target_net(next_states_t).gather(1, next_actions)
                target_q = rewards_t.unsqueeze(1) + (1.0 - dones_t.unsqueeze(1)) * (self.gamma ** self.n_step) * target_q_next
            td_errors = (target_q - current_q).abs()
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach().squeeze())

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            with torch.no_grad():
                for tparam, pparam in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tparam.data.copy_(0.005 * pparam.data + 0.995 * tparam.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()