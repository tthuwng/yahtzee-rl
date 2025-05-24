import os
import random
import math
from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def sample_noise(self) -> None:
        epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device)
        epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQN(nn.Module):
    """Dueling DQN with optional noisy layers."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        use_noisy: bool = False,
    ) -> None:
        super().__init__()
        self.use_noisy = use_noisy
        linear = NoisyLinear if use_noisy else nn.Linear

        self.feature = nn.Sequential(
            linear(state_size, hidden_size),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            linear(hidden_size, hidden_size),
            nn.ReLU(),
            linear(hidden_size, 1),
        )

        self.adv_stream = nn.Sequential(
            linear(hidden_size, hidden_size),
            nn.ReLU(),
            linear(hidden_size, action_size),
        )

    def sample_noise(self) -> None:
        if not self.use_noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.feature(state)
        value = self.value_stream(feat)
        adv = self.adv_stream(feat)
        q = value + adv - adv.mean(1, keepdim=True)
        return q


class YahtzeeAgent:
    """Basic DQN agent for Yahtzee."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 2048,
        gamma: float = 0.97,
        learning_rate: float = 3e-4,
        target_update: int = 50,
        device: str = "auto",
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.9995,
        hidden_size: int = 128,
        use_noisy: bool = False,
        use_amp: bool = False,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.use_amp = use_amp
        self.scaler: Optional[GradScaler] = GradScaler() if use_amp else None

        # Handle device selection
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.learn_steps = 0
        self.training_mode = True
        self.latest_loss: Optional[float] = None
        self.metrics: Dict[str, float] = {}

        # Networks
        self.policy_net = DQN(state_size, action_size, hidden_size, use_noisy).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size, use_noisy).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.use_noisy = use_noisy

        for param in self.target_net.parameters():
            param.requires_grad = False

        # Epsilon-greedy exploration
        self.epsilon = 1.0
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        # Replay buffer
        self.buffer: List[Transition] = []

    def train(self) -> None:
        self.training_mode = True
        self.policy_net.train()
        self.target_net.train()

    def eval(self) -> None:
        self.training_mode = False
        self.policy_net.eval()
        self.target_net.eval()

    def select_action(
        self, state_vec: np.ndarray, valid_actions: List[int]
    ) -> int:
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_t = (
            torch.from_numpy(state_vec)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        q_values = self.policy_net(state_t).squeeze(0)

        # Mask invalid actions
        mask = torch.full(
            (self.action_size,), float("-inf"), device=self.device
        )
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
    ) -> float:
        if self.use_noisy:
            self.policy_net.sample_noise()
            self.target_net.sample_noise()
        # Store transitions
        for s, a, r, ns, d in zip(
            states, actions, rewards, next_states, dones
        ):
            self.buffer.append(Transition(s, a, r, ns, d))
            if len(self.buffer) > 10000:  # Simple buffer size limit
                self.buffer.pop(0)

        if len(self.buffer) < self.batch_size:
            self.latest_loss = None
            self.metrics = {"loss": 0.0, "q_value_mean": 0.0}
            return 0.0

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states_t = (
            torch.from_numpy(np.stack([t.state for t in batch]))
            .float()
            .to(self.device)
        )
        actions_t = torch.tensor(
            [t.action for t in batch], dtype=torch.long
        ).to(self.device)
        rewards_t = torch.tensor(
            [t.reward for t in batch], dtype=torch.float
        ).to(self.device)
        next_states_t = (
            torch.from_numpy(np.stack([t.next_state for t in batch]))
            .float()
            .to(self.device)
        )
        dones_t = torch.tensor(
            [t.done for t in batch], dtype=torch.float
        ).to(self.device)

        # Current Q-values and targets
        with autocast(enabled=self.use_amp):
            current_q = self.policy_net(states_t).gather(
                1, actions_t.unsqueeze(1)
            )

            with torch.no_grad():
                next_q = self.target_net(next_states_t).max(1)[0]
                target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

            loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Update target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

        # Store metrics
        with torch.no_grad():
            q_values = self.policy_net(states_t)
            self.metrics = {
                "loss": loss.item(),
                "q_value_mean": q_values.mean().item(),
            }

        self.latest_loss = loss.item()
        return self.latest_loss

    @torch.no_grad()
    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        state_t = (
            torch.from_numpy(state_vec)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        return self.policy_net(state_t).squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
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
                self.optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )
            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
            if "learn_steps" in checkpoint:
                self.learn_steps = checkpoint["learn_steps"]
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics
