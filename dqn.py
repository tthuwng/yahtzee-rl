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


class DQN(nn.Module):
    """Enhanced DQN with higher capacity and additional residual block."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        # Store sizes for reference
        self.state_size = state_size
        self.action_size = action_size

        # Increase hidden dim from 512 to 768, reduce dropout from 0.05 to 0.02
        hidden_dim = 768
        dropout_rate = 0.02

        # Feature extraction layers
        self.input_layer = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )

        # Residual blocks (3 total now)
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(3)
            ]
        )

        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 3),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 3),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 3, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 3),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 3),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 3, action_size),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"Expected input of size {self.state_size}, got {state.shape[-1]}"
            )

        x = self.input_layer(state)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    """Optimized replay buffer with higher alpha/beta for stronger prioritization."""

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.7,  # increased
        beta: float = 0.5,   # increased
        device: str = "cuda",
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.002  # keep as is or adjust if needed
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

        # Convert numpy arrays to tensors
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
    """Enhanced DQN agent with n-step returns, updated gamma, LR, and slower epsilon decay."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int = 2048,
        gamma: float = 0.99,       # raised from 0.97
        learning_rate: float = 2e-4,  # lowered from 3e-4
        target_update: int = 50,
        device: str = "cuda",
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.9992,  # slower decay
        n_step: int = 3,               # multi-step returns
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

        # N-step storage: for each environment we'll keep a short queue of (state,action,...) until we can pop it.
        # key = env index, value = deque of partial transitions
        self.n_step_buffers = {}

        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        for param in self.target_net.parameters():
            param.requires_grad = False

        # Exploration params
        self.epsilon = 1.0
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Replay buffer with updated alpha/beta
        self.buffer = PrioritizedReplayBuffer(
            capacity=100000,
            alpha=0.7,
            beta=0.5,
            device=device,
        )

        # Optimizer + scheduler
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            amsgrad=True,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20000, eta_min=1e-5
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

        if self.training_mode:
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
            for i, (action, valid_actions) in enumerate(zip(actions, valid_actions_list)):
                if action is None:
                    mask = torch.full(
                        (self.action_size,), float("-inf"), device=self.device
                    )
                    mask[valid_actions] = 0
                    masked_q = q_values[i] + mask
                    actions[i] = masked_q.argmax().item()

        return actions

    def _push_n_step_transition(
        self,
        env_idx: int,
        final_next_state: np.ndarray,
        final_done: bool
    ) -> None:
        """
        After collecting n-step transitions in the local buffer for env_idx,
        compute the multi-step return and push to replay buffer.
        """
        buffer = self.n_step_buffers[env_idx]
        # The first transition in the queue is the one we are finalizing
        first = buffer[0]
        # sum of discounted rewards from first step onwards
        total_reward = 0.0
        gamma_multiplier = 1.0
        for (s, a, r) in [(t.state, t.action, t.reward) for t in buffer]:
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
        if self.training_mode and random.random() < self.epsilon:
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
        env_indices: List[int] = None
    ) -> float:
        """
        Multi-step version: accumulate transitions for each env in n_step_buffers.
        When we have n transitions or the env is done, push to replay.
        """
        if env_indices is None:
            # fallback if not provided
            env_indices = list(range(len(states)))

        for i, (s, a, r, ns, d) in enumerate(
            zip(states, actions, rewards, next_states, dones)
        ):
            env_idx = env_indices[i]
            if env_idx not in self.n_step_buffers:
                from collections import deque
                self.n_step_buffers[env_idx] = deque(maxlen=self.n_step)

            # Add the new single-step to the buffer
            self.n_step_buffers[env_idx].append(Transition(s, a, r, ns, d, 1.0))

            # If we have n steps, finalize the earliest transition
            if len(self.n_step_buffers[env_idx]) == self.n_step:
                self._push_n_step_transition(env_idx, ns, d)

            # If done, flush everything left for that env
            if d:
                while len(self.n_step_buffers[env_idx]) > 0:
                    self._push_n_step_transition(env_idx, ns, d)

        # We can only learn if there's enough data
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample from replay buffer
        states_t, actions_t, rewards_t, next_states_t, dones_t, weights, indices = (
            self.buffer.sample(self.batch_size, self.device)
        )

        with torch.cuda.amp.autocast():
            current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
                target_q = rewards_t.unsqueeze(1) + (1 - dones_t.unsqueeze(1)) * self.gamma * next_q

            td_errors = (target_q - current_q).abs()
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach().squeeze())

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
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    @torch.no_grad()
    def get_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        state_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
        return self.policy_net(state_t).squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
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