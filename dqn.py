import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BigDQN(nn.Module):
    """network with batch normalization."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        # first block - reduced size
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=True)
        self.drop1 = nn.Dropout(0.1)  # Less dropout

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, track_running_stats=True)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, track_running_stats=True)
        self.drop3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(128, action_size)
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)

        x = self.fc4(x)
        return x

    def set_evaluation_mode(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = True

    def set_training_mode(self):
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = True


class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class YahtzeeAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 512,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        target_update: int = 500,
        use_boltzmann: bool = True,
        device: str = "cuda",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.use_boltzmann = use_boltzmann
        self.device = torch.device(device)
        self.learn_steps = 0

        self.policy_net = BigDQN(state_size, action_size).to(self.device)
        self.target_net = BigDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.use_boltzmann:
            self.temperature = 1.0
            self.min_temperature = 0.1
            self.temperature_decay = 0.999
        else:
            self.epsilon = 1.0
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.999

        self.buffer = ReplayBuffer(capacity=100000)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, valid_actions: List[int]) -> int:
        """boltzmann or epsilon-greedy exploration for training."""
        if self.use_boltzmann:
            return self._boltzmann_action(state_vec, valid_actions)
        else:
            return self._epsilon_greedy_action(state_vec, valid_actions)

    @torch.no_grad()
    def select_action_greedy(
        self, state_vec: np.ndarray, valid_actions: List[int]
    ) -> int:
        """
        pure greedy selection among valid_actions (used for testing).
        ignores temperature/epsilon.
        """
        self.policy_net.set_evaluation_mode()
        state_t = torch.tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        qvals = self.policy_net(state_t).squeeze(0)  # shape [action_size]

        # Mask out invalid actions
        mask = torch.full((self.action_size,), float("-inf"), device=self.device)
        mask[valid_actions] = 0
        qvals = qvals + mask
        return qvals.argmax().item()

    @torch.no_grad()
    def _boltzmann_action(self, state_vec: np.ndarray, valid_actions: List[int]) -> int:
        self.policy_net.set_evaluation_mode()
        state_t = torch.tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        qvals = self.policy_net(state_t).squeeze(0)

        mask = torch.full((self.action_size,), float("-inf"), device=self.device)
        mask[valid_actions] = 0
        qvals = qvals + mask

        # softmax with temp
        probs = torch.softmax(qvals / max(self.temperature, 1e-8), dim=0)
        action_idx = torch.multinomial(probs, 1).item()
        return action_idx

    @torch.no_grad()
    def _epsilon_greedy_action(
        self, state_vec: np.ndarray, valid_actions: List[int]
    ) -> int:
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            self.policy_net.set_evaluation_mode()
            state_t = torch.tensor(
                state_vec, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            qvals = self.policy_net(state_t).squeeze(0)
            mask = torch.full((self.action_size,), float("-inf"), device=self.device)
            mask[valid_actions] = 0
            qvals = qvals + mask
            return qvals.argmax().item()

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """stores transition & performs a double dqn update if enough data."""
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample from replay
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # set networks to appropriate modes
        self.policy_net.set_training_mode()
        self.target_net.set_evaluation_mode()

        # current Q
        qvals = self.policy_net(states_t)
        current_q = qvals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # double DQN: next action from policy_net, next Q from target_net
        with torch.no_grad():
            self.policy_net.set_evaluation_mode()  # Important for next action selection
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q_target = (
                self.target_net(next_states_t)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q_target

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay exploration
        if self.use_boltzmann:
            self.temperature = max(
                self.min_temperature, self.temperature * self.temperature_decay
            )
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    @torch.no_grad()
    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        """get Q-values for all actions in the given state."""
        self.policy_net.set_evaluation_mode()
        state_t = torch.tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()
        return q_values
