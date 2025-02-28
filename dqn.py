import math
import os
import random
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "td_error")
)


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration without epsilon greedy"""

    def __init__(
        self, in_features: int, out_features: int, std_init: float = 0.5
    ) -> None:
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Mean weights
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))

        # Std deviation weights
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # Initialize parameters
        self.reset_parameters()

        # Noise buffers
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        self.sample_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Create scaled noise for factorized Gaussian noise"""
        noise = torch.randn(size)
        return noise.sign().mul(noise.abs().sqrt())

    def sample_noise(self) -> None:
        """Sample and store noise for forward pass"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise injection"""
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class ResidualBlock(nn.Module):
    """Enhanced residual block with gating mechanism"""

    def __init__(self, channels: int, dropout_rate: float = 0.02) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout_rate),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout_rate),
        )

        # Gating mechanism
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        gate_vals = self.gate(x)
        return residual + out * gate_vals


class DQN(nn.Module):
    """Enhanced DQN with residual blocks and noisy exploration."""

    def __init__(
        self, state_size: int, action_size: int, use_noisy: bool = True
    ) -> None:
        super().__init__()

        # Store sizes for reference
        self.state_size = state_size
        self.action_size = action_size
        self.use_noisy = use_noisy

        # Use 512 hidden dim to match saved model
        hidden_dim = 512
        dropout_rate = 0.02

        # Feature extraction layers
        self.input_layer = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate)
                for _ in range(3)  # Increased from 2 to 3
            ]
        )

        # Dueling architecture
        if use_noisy:
            # Noisy advantage stream
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout_rate),
                NoisyLinear(hidden_dim // 2, action_size),
            )

            # Noisy value stream
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout_rate),
                NoisyLinear(hidden_dim // 2, 1),
            )
        else:
            # Standard advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, action_size),
            )

            # Standard value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, NoisyLinear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"Expected input of size {self.state_size}, got {state.shape[-1]}"
            )

        # Use automatic mixed precision for A100
        with torch.cuda.amp.autocast():
            x = self.input_layer(state)

            for block in self.res_blocks:
                x = block(x)

            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def sample_noise(self) -> None:
        """Sample new noise for exploration"""
        if not self.use_noisy:
            return

        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


class PrioritizedReplayBuffer:
    """Optimized replay buffer with higher alpha/beta for stronger prioritization."""

    def __init__(
        self,
        capacity: int = 200000,
        alpha: float = 0.7,  # increased
        beta: float = 0.5,  # increased
        device: str = "cuda",
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.002
        self.buffer: List[Transition] = []
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
        n_samples = min(len(self.buffer), self.capacity)
        probs = self.priorities[:n_samples] / self.priorities[:n_samples].sum()

        indices = torch.multinomial(probs, batch_size, replacement=True)

        # Importance sampling weights
        weights = (n_samples * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[idx.item()] for idx in indices]

        states = torch.from_numpy(np.stack([t.state for t in batch])).float().to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float).to(device)
        next_states = (
            torch.from_numpy(np.stack([t.next_state for t in batch])).float().to(device)
        )
        dones = torch.tensor([t.done for t in batch], dtype=torch.float).to(device)
        weights = weights.to(device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(
        self, indices: torch.Tensor, new_errors: torch.Tensor
    ) -> None:
        indices = indices.to(self.device)
        new_errors = new_errors.to(self.device)
        priorities = (new_errors.abs() + self.eps) ** self.alpha
        self.priorities[indices] = priorities

    def __len__(self) -> int:
        return len(self.buffer)


class YahtzeeAgent:
    """Enhanced DQN agent with n-step returns, updated gamma, LR, and noisy exploration."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 2048,  # Increased for A100
        gamma: float = 0.997,
        learning_rate: float = 1e-4,
        target_update: int = 50,
        device: str = "cuda",
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.9992,
        n_step: int = 5,
        use_noisy: bool = True,  # Use noisy networks instead of epsilon-greedy
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.device = torch.device(device)
        self.learn_steps = 0
        self.training_mode = True
        self.n_step = n_step
        self.use_noisy = use_noisy
        self.latest_loss: Optional[float] = None

        # Training metrics dictionary
        self.metrics: Dict[str, float] = {}

        # N-step storage
        self.n_step_buffers = {}

        # Networks
        self.policy_net = DQN(state_size, action_size, use_noisy=use_noisy).to(
            self.device
        )
        self.target_net = DQN(state_size, action_size, use_noisy=use_noisy).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        for param in self.target_net.parameters():
            param.requires_grad = False

        # Only use epsilon if not using noisy networks
        self.epsilon = 1.0 if not use_noisy else min_epsilon
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Mixed precision scaler for A100
        self.scaler = torch.cuda.amp.GradScaler()

        # Replay buffer with updated capacity
        self.buffer = PrioritizedReplayBuffer(
            capacity=300000,  # increased from 200000
            alpha=0.7,
            beta=0.5,
            device=device,
        )

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            amsgrad=True,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=40000, eta_min=1e-5
        )

    def train(self) -> None:
        self.training_mode = True
        self.policy_net.train()
        self.target_net.train()

    def eval(self) -> None:
        self.training_mode = False
        self.policy_net.eval()
        self.target_net.eval()

    @torch.no_grad()
    def select_actions_batch(
        self, state_vecs: np.ndarray, valid_actions_list: List[List[int]]
    ) -> List[int]:
        batch_size = len(state_vecs)
        actions = []

        # If using noisy networks, sample fresh noise for exploration
        if self.use_noisy and self.training_mode:
            self.policy_net.sample_noise()

        # Only do random actions if not using noisy networks
        if self.training_mode and not self.use_noisy:
            random_mask = np.random.random(batch_size) < self.epsilon
            for i, should_random in enumerate(random_mask):
                if should_random:
                    actions.append(random.choice(valid_actions_list[i]))
                else:
                    actions.append(None)
        else:
            actions = [None] * batch_size

        if any(a is None for a in actions):
            states_t = torch.from_numpy(state_vecs).float().to(self.device)
            q_values = self.policy_net(states_t)
            for i, (action, valid_actions) in enumerate(
                zip(actions, valid_actions_list)
            ):
                if action is None:
                    mask = torch.full(
                        (self.action_size,), float("-inf"), device=self.device
                    )
                    mask[valid_actions] = 0
                    masked_q = q_values[i] + mask
                    actions[i] = masked_q.argmax().item()

        return actions

    def _push_n_step_transition(
        self, env_idx: int, final_next_state: np.ndarray, final_done: bool
    ) -> None:
        buffer = self.n_step_buffers[env_idx]
        first = buffer[0]
        total_reward = 0.0
        gamma_multiplier = 1.0
        for s, a, r in [(t.state, t.action, t.reward) for t in buffer]:
            total_reward += r * gamma_multiplier
            gamma_multiplier *= self.gamma

        self.buffer.push(
            first.state,
            first.action,
            total_reward,
            final_next_state,
            final_done,
        )
        buffer.popleft()

    def select_action(self, state_vec: np.ndarray, valid_actions: List[int]) -> int:
        # Sample noise for exploration if using noisy networks
        if self.use_noisy and self.training_mode:
            self.policy_net.sample_noise()

        # Only use epsilon-greedy if not using noisy networks
        if self.training_mode and not self.use_noisy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        q_values = self.policy_net(state_t).squeeze(0)

        mask = torch.full((self.action_size,), float("-inf"), device=self.device)
        mask[valid_actions] = 0
        q_values = q_values + mask
        return q_values.argmax().item()

    def train_step_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
        env_indices: List[int] = None,
    ) -> float:
        if env_indices is None:
            env_indices = list(range(len(states)))

        for i, (s, a, r, ns, d) in enumerate(
            zip(states, actions, rewards, next_states, dones)
        ):
            env_idx = env_indices[i]
            if env_idx not in self.n_step_buffers:
                from collections import deque

                self.n_step_buffers[env_idx] = deque(maxlen=self.n_step)

            self.n_step_buffers[env_idx].append(Transition(s, a, r, ns, d, 1.0))

            if len(self.n_step_buffers[env_idx]) == self.n_step:
                self._push_n_step_transition(env_idx, ns, d)

            if d:
                while len(self.n_step_buffers[env_idx]) > 0:
                    self._push_n_step_transition(env_idx, ns, d)

        if len(self.buffer) < self.batch_size:
            self.latest_loss = None
            self.metrics = {"loss": 0.0, "q_value_mean": 0.0, "q_value_max": 0.0}
            return 0.0

        states_t, actions_t, rewards_t, next_states_t, dones_t, weights, indices = (
            self.buffer.sample(self.batch_size, self.device)
        )

        # Use mixed precision for A100
        with torch.cuda.amp.autocast():
            # Current Q-values
            current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))

            # Next Q-values with double Q-learning
            with torch.no_grad():
                next_actions = self.policy_net(next_states_t).argmax(
                    dim=1, keepdim=True
                )
                next_q = self.target_net(next_states_t).gather(1, next_actions)
                target_q = (
                    rewards_t.unsqueeze(1)
                    + (1 - dones_t.unsqueeze(1)) * (self.gamma**self.n_step) * next_q
                )

            # Calculate TD errors and loss
            td_errors = (target_q - current_q).abs()
            loss = (
                weights * F.smooth_l1_loss(current_q, target_q, reduction="none")
            ).mean()

        # Optimize with mixed precision
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update priorities in buffer
        self.buffer.update_priorities(indices, td_errors.detach().squeeze())

        # Soft target network update
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            with torch.no_grad():
                for target_param, policy_param in zip(
                    self.target_net.parameters(), self.policy_net.parameters()
                ):
                    target_param.data.copy_(
                        0.005 * policy_param.data + 0.995 * target_param.data
                    )

        self.scheduler.step()

        # Only decay epsilon if not using noisy networks
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Store metrics
        with torch.no_grad():
            q_values = self.policy_net(states_t)
            self.metrics = {
                "loss": loss.item(),
                "q_value_mean": q_values.mean().item(),
                "q_value_max": q_values.max().item(),
                "td_error_mean": td_errors.mean().item(),
                "learning_rate": self.scheduler.get_last_lr()[0],
            }

        self.latest_loss = loss.item()
        return self.latest_loss

    @torch.no_grad()
    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        state_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        return self.policy_net(state_t).squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
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
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
            if "learn_steps" in checkpoint:
                self.learn_steps = checkpoint["learn_steps"]
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)

    def get_metrics(self) -> Dict[str, float]:
        """Return current training metrics"""
        return self.metrics
