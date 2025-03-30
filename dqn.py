import os
import random
from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class DQN(nn.Module):
    """Basic DQN model for Yahtzee."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Simple feed-forward network
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class YahtzeeAgent:
    """Basic DQN agent for Yahtzee."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        target_update: int = 10,
        device: str = "cuda",
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.device = torch.device(device)
        self.learn_steps = 0
        self.training_mode = True
        self.latest_loss: Optional[float] = None
        self.metrics: Dict[str, float] = {}

        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

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

        # Current Q-values
        current_q = self.policy_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        )

        # Next Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        # Calculate loss
        loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
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
