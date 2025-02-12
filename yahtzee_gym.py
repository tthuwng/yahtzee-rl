from enum import Enum
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from encoder import StateEncoder
from env import NUM_ACTIONS, Action, ActionType, YahtzeeCategory, YahtzeeEnv


class RewardStrategy(Enum):
    """Different strategies for shaping the reward signal."""

    STANDARD = "standard"  # Original reward function
    STRATEGIC = "strategic"  # Use env's strategic reward shaping
    NORMALIZED = "normalized"  # Normalize rewards to [-1, 1]
    SPARSE = "sparse"  # Only give reward at end of episode
    POTENTIAL = "potential"  # Use potential-based shaping


class YahtzeeGymEnv(gym.Env):
    """Gymnasium-style environment for Yahtzee.

    Observation Space:
        Box(22,) or Box(23,) depending on use_opponent_value:
        - [0]: rolls_left (normalized 0-1)
        - [1:7]: dice counts (normalized 0-1)
        - [7:20]: category flags (0 or 1)
        - [20:22]: upper/lower scores (normalized 0-1)
        - [22]: opponent value (optional, normalized 0-1)

    Action Space:
        Discrete(46):
        - [0]: roll all dice
        - [1:33]: hold dice combinations
        - [33:46]: score categories
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        use_opponent_value: bool = False,
        render_mode: Optional[str] = None,
        reward_strategy: RewardStrategy = RewardStrategy.STRATEGIC,
    ):
        super().__init__()

        # Create underlying environment and encoder
        self.env = YahtzeeEnv()
        self.encoder = StateEncoder(use_opponent_value=use_opponent_value)
        self.reward_strategy = reward_strategy

        # For potential-based shaping
        self._last_potential: Optional[float] = None

        # Define action space (46 possible actions)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Define observation space
        obs_size = 23 if use_opponent_value else 22
        obs_low = 0.0
        obs_high = 1.0
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(obs_size,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.use_opponent_value = use_opponent_value

        # For rendering
        self.last_action: Optional[Action] = None
        self.last_reward: float = 0.0

    def _compute_potential(self) -> float:
        """Compute potential function for reward shaping."""
        state = self.env.state
        total_score = sum(
            score for score in state.score_sheet.values() if score is not None
        )
        moves_left = sum(1 for score in state.score_sheet.values() if score is None)
        avg_per_move = total_score / (13 - moves_left) if moves_left < 13 else 0
        potential = avg_per_move * moves_left
        return potential / 100  # Normalize to reasonable range

    def _shape_reward(
        self, reward: float, done: bool, action: Action, base_score: int
    ) -> float:
        """Apply reward shaping based on selected strategy."""
        if self.reward_strategy == RewardStrategy.STANDARD:
            return reward

        elif self.reward_strategy == RewardStrategy.STRATEGIC:
            # Use the environment's strategic reward shaping for scoring actions
            if action.kind == ActionType.SCORE:
                return self.env.calc_strategic_reward(action.data, base_score)
            return reward

        elif self.reward_strategy == RewardStrategy.NORMALIZED:
            # Normalize to [-1, 1] range (assuming max reward ~50)
            return np.clip(reward / 50, -1, 1)

        elif self.reward_strategy == RewardStrategy.SPARSE:
            # Only give reward at end of episode
            return reward if done else 0.0

        elif self.reward_strategy == RewardStrategy.POTENTIAL:
            # Potential-based shaping
            current_potential = self._compute_potential()
            if self._last_potential is None:
                shaped_reward = reward
            else:
                shaped_reward = reward + (
                    0.99 * current_potential - self._last_potential
                )
            self._last_potential = current_potential
            return shaped_reward

        return reward

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset underlying environment
        game_state = self.env.reset()

        # Reset potential
        self._last_potential = None

        # Get opponent value for encoding
        opp_value = 0.5 if self.use_opponent_value else 0.0

        # Encode state
        obs = self.encoder.encode(game_state, opponent_value=opp_value)

        # Reset render info
        self.last_action = None
        self.last_reward = 0.0

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take action and return (obs, reward, terminated, truncated, info)."""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # Get action object
        action_obj = self.env.idx_to_action[action]

        # Calculate base score if scoring action
        base_score = 0
        if action_obj.kind == ActionType.SCORE:
            base_score = self.env.calc_score(
                action_obj.data, self.env.state.current_dice
            )

        # Take action in underlying environment
        game_state, reward, done, info = self.env.step(action)

        # Apply reward shaping
        shaped_reward = self._shape_reward(reward, done, action_obj, base_score)

        # Store for rendering
        self.last_action = action_obj
        self.last_reward = shaped_reward

        # Get opponent value for encoding
        opp_value = 0.5 if self.use_opponent_value else 0.0

        # Encode next state
        obs = self.encoder.encode(game_state, opponent_value=opp_value)

        # Add reward info
        info["original_reward"] = reward
        info["shaped_reward"] = shaped_reward
        if base_score > 0:
            info["base_score"] = base_score

        # Gymnasium requires truncated flag (always False for Yahtzee)
        truncated = False

        return obs, shaped_reward, done, truncated, info

    def render(self) -> Optional[str]:
        """Render the current state."""
        if self.render_mode is None:
            return None

        # Get current game state
        state = self.env.state

        # Format dice display
        dice_vals = state.current_dice
        dice_str = " ".join(str(d) if d > 0 else "-" for d in dice_vals)
        rolls = state.rolls_left

        # Build output
        lines = []
        lines.append(f"Dice: [{dice_str}] (rolls left: {rolls})")

        if self.last_action is not None:
            action_str = self._format_action(self.last_action)
            lines.append(f"\nLast action: {action_str}")
            lines.append(f"Reward: {self.last_reward:.1f}")

        lines.append("\nScore sheet:")
        for cat in YahtzeeCategory:
            score = state.score_sheet[cat]
            score_str = str(score) if score is not None else "-"
            lines.append(f"{cat.name}: {score_str}")

        bonus = self.env.calc_upper_bonus()
        if bonus > 0:
            lines.append(f"\nUpper Bonus: +{bonus}")

        output = "\n".join(lines)

        if self.render_mode == "human":
            print(output)
            return None
        else:
            return output

    def _format_action(self, action: Action) -> str:
        """Format action for display."""
        if action.kind == ActionType.ROLL:
            return "ROLL all dice"
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                held_str = ", ".join(str(pos) for pos in held)
                return f"Hold dice at positions {held_str}"
            else:
                return "ROLL all dice"
        else:
            points = self.env.calc_score(action.data, self.env.state.current_dice)
            return f"Score {action.data.name} for {points} points"

    def close(self) -> None:
        """Clean up environment. Nothing to do for Yahtzee."""
        pass
