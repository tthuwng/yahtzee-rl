import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "td_error")
)


class DQN(nn.Module):
    """Deep Q-Network for Yahtzee with dueling architecture."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        # Feature extraction layers with residual connections
        self.input_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        # Residual blocks
        self.res_block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # Value stream for state value estimation
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

        # Advantage stream for action advantages
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_size),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass with residual connections
        features = self.input_layer(x)

        # First residual block
        res1 = self.res_block1(features)
        features = features + res1

        # Second residual block
        res2 = self.res_block2(features)
        features = features + res2

        # Output processing
        features = self.output_layer(features)

        # Dueling streams
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvalues = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues


class PrioritizedReplayBuffer:
    """
    A simple prioritized replay buffer.
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.priorities: List[float] = []
        self.alpha = alpha
        self.pos = 0
        self.eps = 1e-3

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
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer!")
        priorities_np = np.array(self.priorities, dtype=np.float32)
        prob = priorities_np / priorities_np.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)
        batch = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([t.done for t in batch])
        return (states, actions, rewards, next_states, dones, indices)

    def update_priorities(self, indices: List[int], new_errors: np.ndarray) -> None:
        for idx, err in zip(indices, new_errors):
            self.priorities[idx] = (abs(err) + self.eps) ** self.alpha

    def __len__(self) -> int:
        return len(self.buffer)


class YahtzeeAgent:
    """DQN agent with epsilon-greedy exploration."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 2e-4,
        target_update: int = 100,
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

        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Exploration schedule
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9999  # Slower decay

        # Use prioritized replay buffer instead of deque
        self.buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)

        # Optimizer with AdamW
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            amsgrad=True,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10000, T_mult=2, eta_min=1e-5
        )

    def train(self) -> None:
        self.training_mode = True
        self.policy_net.train()
        self.target_net.train()

    def eval(self) -> None:
        self.training_mode = False
        self.policy_net.eval()
        self.target_net.eval()

    def select_action(self, state_vec: np.ndarray, valid_actions: List[int]) -> int:
        """Epsilon-greedy action selection."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).to(self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0)

            # Mask invalid actions
            mask = torch.full((self.action_size,), float("-inf"), device=self.device)
            mask[valid_actions] = 0
            q_values = q_values + mask

            if self.training_mode and random.random() < self.epsilon:
                # Bias towards scoring actions when exploring
                score_actions = [
                    a for a in valid_actions if a >= 33
                ]  # Scoring actions start at index 33
                if score_actions and random.random() < 0.3:  # 30% chance to force score
                    return random.choice(score_actions)
                return random.choice(valid_actions)
            else:
                return q_values.argmax().item()

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        # Store transition with default td_error of 1.0
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample batch from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices = self.buffer.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values with Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = (
                rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
            )

        # Compute loss with gradient clipping
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update learning rate scheduler every 100 steps
        if self.learn_steps % 100 == 0:
            self.scheduler.step()

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

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update priorities based on new TD errors
        with torch.no_grad():
            td_errors = (target_q - current_q).abs().squeeze(1).cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        return loss.item()

    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            return self.policy_net(state_t).squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights with backward compatibility."""
        print(f"Loading model from: {path}")
        try:
            # Try loading with new architecture first
            state_dict = torch.load(path, map_location=self.device)
            if isinstance(state_dict, dict) and "policy_net" in state_dict:
                # New checkpoint format
                self.policy_net.load_state_dict(state_dict["policy_net"])
                self.target_net.load_state_dict(state_dict["target_net"])
                self.optimizer.load_state_dict(state_dict["optimizer"])
                print("Successfully loaded checkpoint format")
                return

            # Try loading as direct state dict
            try:
                self.policy_net.load_state_dict(state_dict)
                self.target_net.load_state_dict(state_dict)
                print("Successfully loaded state dict format")
                return
            except:
                pass

            print("Converting legacy model format...")
            # Convert legacy model with different dimensions
            new_state = {}

            # Helper function to resize tensors
            def resize_tensor(tensor: torch.Tensor, new_shape: tuple) -> torch.Tensor:
                if len(tensor.shape) == 1:  # For bias terms
                    # For 1D tensors, we'll either truncate or pad with zeros
                    result = torch.zeros(new_shape[0], device=tensor.device)
                    min_size = min(tensor.shape[0], new_shape[0])
                    result[:min_size] = tensor[:min_size]
                    return result
                elif len(tensor.shape) == 2:  # For weight matrices
                    # For 2D tensors, we'll use interpolation
                    tensor = tensor.float()  # Ensure float type for interpolation
                    tensor = tensor.unsqueeze(0).unsqueeze(
                        0
                    )  # Add batch and channel dims
                    resized = torch.nn.functional.interpolate(
                        tensor, size=new_shape, mode="bilinear", align_corners=True
                    )
                    return resized.squeeze(0).squeeze(0)
                return tensor

            # Map legacy features to new architecture with resizing
            for name, param in state_dict.items():
                if name in self.policy_net.state_dict():
                    target_shape = self.policy_net.state_dict()[name].shape
                    new_state[name] = resize_tensor(param, target_shape)

            # Load converted state dict
            self.policy_net.load_state_dict(new_state)
            self.target_net.load_state_dict(new_state)
            print("Successfully loaded and converted legacy model format")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
