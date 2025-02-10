import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "td_error")
)


class DQN(nn.Module):
    """Optimized DQN for faster training."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        # Streamlined architecture with efficient layer sizes
        self.input_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),  # Changed to ReLU for better stability
            nn.LayerNorm(256),
        )

        # Single efficient residual block
        self.res_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        # Efficient value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_size),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Efficient forward pass
        features = self.input_layer(x)

        # Single residual connection
        res = self.res_block(features)
        features = features + res

        # Dueling streams
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine streams
        qvalues = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues


class PrioritizedReplayBuffer:
    """Optimized replay buffer for faster training."""

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str = "cuda",
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.002  # Faster beta annealing
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

        # Sample with priorities
        probs = self.priorities[:n_samples] / self.priorities[:n_samples].sum()
        indices = torch.multinomial(probs, batch_size, replacement=True)

        # Calculate importance sampling weights
        weights = (n_samples * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Create batch from sampled indices
        batch = [self.buffer[idx.item()] for idx in indices]

        # Convert numpy arrays to tensors and move to correct device
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
        # Move tensors to the same device as priorities
        indices = indices.to(self.device)
        new_errors = new_errors.to(self.device)
        priorities = (new_errors.abs() + self.eps) ** self.alpha
        self.priorities[indices] = priorities

    def __len__(self) -> int:
        return len(self.buffer)


class YahtzeeAgent:
    """Optimized DQN agent for faster training."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 1024,
        gamma: float = 0.99,
        learning_rate: float = 2e-4,
        target_update: int = 50,
        device: str = "cuda",
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.device = torch.device(device)
        self.learn_steps = 0
        self.training_mode = True

        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        for param in self.target_net.parameters():
            param.requires_grad = False

        # Faster exploration decay
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995

        # Smaller buffer for faster training with correct device
        self.buffer = PrioritizedReplayBuffer(capacity=50000, device=device)

        # Optimizer with higher learning rate
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            amsgrad=True,
        )

        # Fixed scheduler using proper import
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20000, eta_min=1e-5
        )

    def train(self) -> None:
        """Set networks to training mode."""
        self.training_mode = True
        self.policy_net.train()
        self.target_net.train()

    def eval(self) -> None:
        """Set networks to evaluation mode."""
        self.training_mode = False
        self.policy_net.eval()
        self.target_net.eval()

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, valid_actions: List[int]) -> int:
        """Epsilon-greedy action selection with GPU acceleration."""
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Convert state to tensor and move to correct device
        state_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        q_values = self.policy_net(state_t).squeeze(0)

        # Mask invalid actions
        mask = torch.full((self.action_size,), float("-inf"), device=self.device)
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
    ) -> float:
        # Store transition
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample batch with importance sampling
        states, actions, rewards, next_states, dones, weights, indices = (
            self.buffer.sample(self.batch_size, self.device)
        )

        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values with Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = (
                rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
            )

        # Compute loss with importance sampling
        td_errors = (target_q - current_q).abs()
        loss = (
            weights * F.smooth_l1_loss(current_q, target_q, reduction="none")
        ).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.detach().squeeze())

        # Soft update target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            with torch.no_grad():
                for target_param, policy_param in zip(
                    self.target_net.parameters(), self.policy_net.parameters()
                ):
                    target_param.data.copy_(
                        0.005 * policy_param.data + 0.995 * target_param.data
                    )

        # Update learning rate and epsilon
        self.scheduler.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    @torch.no_grad()
    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in current state."""
        state_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        return self.policy_net(state_t).squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        """Save model with full training state."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model with full training state."""
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
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
