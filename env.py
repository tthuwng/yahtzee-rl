import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from yahtzee_types import Action, ActionType, GameState, YahtzeeCategory

from encoder import NUM_ACTIONS, ActionMapper, StateEncoder


class RewardStrategy(Enum):
    """Different strategies for shaping the reward signal."""

    STANDARD = "standard"
    STRATEGIC = "strategic"
    NORMALIZED = "normalized"
    SPARSE = "sparse"
    POTENTIAL = "potential"


class YahtzeeEnv(gym.Env):
    """Unified Yahtzee environment with gym interface."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        use_opponent_value: bool = False,
        reward_strategy: RewardStrategy = RewardStrategy.STRATEGIC,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.use_opponent_value = use_opponent_value
        self.reward_strategy = reward_strategy
        self.render_mode = render_mode

        # For potential-based shaping
        self._last_potential: Optional[float] = None

        # Initialize encoders and action mapping
        self.encoder = StateEncoder(use_opponent_value=use_opponent_value)
        self.action_mapper = ActionMapper()
        self._opponent_value = 0.5 if use_opponent_value else 0.0

        # Gym Spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        obs_size = 22 if not use_opponent_value else 23
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Initialize game state
        self.state: GameState = GameState(
            current_dice=np.zeros(5, dtype=int),
            rolls_left=3,
            score_sheet={cat: None for cat in YahtzeeCategory},
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> GameState:
        """
        Maintains backward-compatibility with older code:
          env.reset() -> GameState

        Also readies the environment for gym usage (by storing observation in info).
        If you need gym-style returns, see info["obs"] after calling reset.
        """
        if seed is not None:
            super().reset(seed=seed)

        # Reset potential if we're using potential-based shaping
        self._last_potential = None

        self.state = GameState(
            current_dice=np.zeros(5, dtype=int),
            rolls_left=3,
            score_sheet={cat: None for cat in YahtzeeCategory},
        )

        # Return the game state to keep old code working
        return self.state

    # --------------------------------------------------------------------------------
    # Old style step: returns (GameState, reward, done, info).
    # Also we embed the computed observation in info['obs'] if needed.
    # --------------------------------------------------------------------------------
    def step(self, action_idx: int) -> Tuple[GameState, float, bool, dict]:
        """Step the environment with an action index."""
        # Validate
        if action_idx < 0 or action_idx >= NUM_ACTIONS:
            raise ValueError(f"Invalid action index {action_idx}")

        action = self.action_mapper.index_to_action(action_idx)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action_type": action.kind.name}

        # If scoring, we'll check base_score for shaping logic
        base_score = 0

        # ROLL or HOLD
        if action.kind == ActionType.ROLL:
            if self.state.rolls_left <= 0:
                raise ValueError("No rolls left!")
            self._roll_dice(np.zeros(5, dtype=bool))
            info["dice_rolled"] = True

        elif action.kind == ActionType.HOLD:
            if self.state.rolls_left <= 0:
                raise ValueError("No rolls left!")
            if not np.any(self.state.current_dice):
                raise ValueError("Cannot hold empty dice!")
            hold_mask = np.array(action.data, dtype=bool)
            self._roll_dice(hold_mask)
            info["dice_held"] = np.where(hold_mask)[0].tolist()

        elif action.kind == ActionType.SCORE:
            category = action.data
            if self.state.score_sheet[category] is not None:
                raise ValueError(f"Category {category} already filled!")
            if not np.any(self.state.current_dice):
                raise ValueError("Cannot score empty dice!")
            # Compute points
            points = self.calc_score(category, self.state.current_dice)
            base_score = points

            # The environment's default reward is the shaped reward
            shaped = self._shape_reward(
                float(points), action, done=False, base_score=points
            )
            reward = shaped

            # Assign points
            self.state.score_sheet[category] = points
            info["category_scored"] = category.name
            info["points_scored"] = points

            # Check if game is over
            if all(sc is not None for sc in self.state.score_sheet.values()):
                done = True
                bonus = self.calc_upper_bonus()
                reward += bonus  # add bonus to final reward
                info["upper_bonus"] = bonus
                info["final_score"] = (
                    sum(
                        score
                        for score in self.state.score_sheet.values()
                        if score is not None
                    )
                    + bonus
                )

            # Reset dice for next turn
            self.state.current_dice = np.zeros(5, dtype=int)
            self.state.rolls_left = 3

        # Now if the action was ROLL or HOLD, we do standard reward shaping
        else:
            # This shouldn't happen, but let's keep consistent
            shaped = self._shape_reward(0.0, action, done=False, base_score=0)
            reward = shaped

        # If done, do potential shaping one last time if needed
        if done:
            # shape again with final done if needed
            reward = self._shape_reward(
                reward, action, done=True, base_score=base_score
            )

        # Create obs for info
        obs = self.encoder.encode(self.state, opponent_value=self._opponent_value)
        info["obs"] = obs

        return self.state, reward, done, info

    def _roll_dice(self, hold_mask: np.ndarray):
        if self.state.rolls_left <= 0:
            raise ValueError("No rolls left!")
        for i in range(5):
            if not hold_mask[i]:
                self.state.current_dice[i] = random.randint(1, 6)
        self.state.rolls_left -= 1

    def get_valid_actions(self) -> List[int]:
        """List of valid action indices for the current state."""
        return self.action_mapper.get_valid_actions(self.state)

    # --------------------------------------------------------------------------------
    # Score calculations and final scoring
    # --------------------------------------------------------------------------------
    def calc_score(self, category: YahtzeeCategory, dice: np.ndarray) -> int:
        """Calculate the base score for a category."""
        if not np.any(dice > 0):  # no dice shown
            return 0

        counts = np.bincount(dice, minlength=7)[1:]  # skip index 0
        if category == YahtzeeCategory.ONES:
            return counts[0] * 1
        elif category == YahtzeeCategory.TWOS:
            return counts[1] * 2
        elif category == YahtzeeCategory.THREES:
            return counts[2] * 3
        elif category == YahtzeeCategory.FOURS:
            return counts[3] * 4
        elif category == YahtzeeCategory.FIVES:
            return counts[4] * 5
        elif category == YahtzeeCategory.SIXES:
            return counts[5] * 6
        elif category == YahtzeeCategory.THREE_OF_A_KIND:
            if max(counts) >= 3:
                return dice.sum()
            return 0
        elif category == YahtzeeCategory.FOUR_OF_A_KIND:
            if max(counts) >= 4:
                return dice.sum()
            return 0
        elif category == YahtzeeCategory.FULL_HOUSE:
            # check 3-of-a-kind + 2-of-a-kind
            has_three = any(c == 3 for c in counts)
            has_two = any(c == 2 for c in counts)
            return 25 if has_three and has_two else 0
        elif category == YahtzeeCategory.SMALL_STRAIGHT:
            # unique sorted
            sorted_unique = np.unique(dice)
            straights = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
            for st in straights:
                if all(x in sorted_unique for x in st):
                    return 30
            return 0
        elif category == YahtzeeCategory.LARGE_STRAIGHT:
            sorted_unique = np.unique(dice)
            # [1..5] or [2..6]
            if len(sorted_unique) == 5 and (
                all(x in sorted_unique for x in [1, 2, 3, 4, 5])
                or all(x in sorted_unique for x in [2, 3, 4, 5, 6])
            ):
                return 40
            return 0
        elif category == YahtzeeCategory.YAHTZEE:
            if max(counts) == 5:
                return 50
            return 0
        elif category == YahtzeeCategory.CHANCE:
            return dice.sum()
        else:
            return 0

    def calc_upper_bonus(self) -> int:
        """Compute the standard Yahtzee upper bonus if >= 63 in upper categories."""
        upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]
        upper_score = sum(self.state.score_sheet[cat] or 0 for cat in upper_cats)
        return 35 if upper_score >= 63 else 0

    # --------------------------------------------------------------------------------
    # Reward shaping methods
    # --------------------------------------------------------------------------------
    def _shape_reward(
        self, reward: float, action: Action, done: bool, base_score: float
    ) -> float:
        """Apply the reward strategy shaping."""
        if self.reward_strategy == RewardStrategy.STANDARD:
            # no shaping, just keep the environment's raw reward
            return reward

        elif self.reward_strategy == RewardStrategy.STRATEGIC:
            # use the internal strategic approach if scoring
            if action.kind == ActionType.SCORE:
                return self.calc_strategic_reward(action.data, base_score)
            else:
                # ROLL/HOLD yield no direct points but small or 0 shaping
                return reward

        elif self.reward_strategy == RewardStrategy.NORMALIZED:
            # naive normalization
            # assume max single scoring is about 50 for base, clip
            return float(np.clip(reward / 50.0, -1.0, 1.0))

        elif self.reward_strategy == RewardStrategy.SPARSE:
            # only reward at the end
            return reward if done else 0.0

        elif self.reward_strategy == RewardStrategy.POTENTIAL:
            # potential-based shaping
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

    def calc_strategic_reward(
        self, category: YahtzeeCategory, base_score: float
    ) -> float:
        """
        A shaped reward encouraging good Yahtzee tactics.
        We do something similar to the older approach in the original code.
        """
        dice = self.state.current_dice
        counts = np.bincount(dice)[1:] if any(dice) else []
        max_count = max(counts) if len(counts) > 0 else 0

        bonus_reward = 0.0

        # reward if base_score > 0
        if base_score > 0:
            bonus_reward += 2.0
        else:
            # penalty for 0
            bonus_reward -= 8.0

        # encourage bigger combos
        if max_count >= 4 and (
            self.state.score_sheet.get(YahtzeeCategory.YAHTZEE) is None
        ):
            bonus_reward += 10.0
        elif max_count >= 3:
            bonus_reward += 4.0

        # upper categories
        upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]
        upper_score_so_far = sum(self.state.score_sheet[cat] or 0 for cat in upper_cats)
        upper_filled = sum(
            1 for cat in upper_cats if self.state.score_sheet[cat] is not None
        )

        if category in upper_cats:
            # if we got at least 3 of that face
            val_index = upper_cats.index(category)
            face_val = val_index + 1
            if base_score >= face_val * 3:
                bonus_reward += 7.0
            else:
                bonus_reward += 2.0
            # slight scale for bigger faces
            bonus_reward += face_val * 0.3

            # encourage finishing the upper bonus early
            if (upper_score_so_far + base_score) >= 63 and upper_filled < 5:
                bonus_reward += 6.0

        # special combos
        if category == YahtzeeCategory.FULL_HOUSE and base_score == 25:
            bonus_reward += 5.0
        elif category == YahtzeeCategory.SMALL_STRAIGHT and base_score == 30:
            bonus_reward += 6.0
        elif category == YahtzeeCategory.LARGE_STRAIGHT and base_score == 40:
            bonus_reward += 8.0
        elif category == YahtzeeCategory.YAHTZEE and base_score == 50:
            bonus_reward += 15.0

        # penalty for scoring 0 on big combos
        if base_score == 0 and category in [
            YahtzeeCategory.FULL_HOUSE,
            YahtzeeCategory.LARGE_STRAIGHT,
            YahtzeeCategory.YAHTZEE,
        ]:
            bonus_reward -= 5.0

        return base_score + bonus_reward

    def _compute_potential(self) -> float:
        """
        Potential-based shaping function: consider average score so far,
        moves left, etc. We just do a simple approach for demonstration.
        """
        total_score = sum(
            sc for sc in self.state.score_sheet.values() if sc is not None
        )
        moves_left = sum(1 for sc in self.state.score_sheet.values() if sc is None)
        # naive potential
        if moves_left < 13:
            avg_per_move = (
                total_score / (13 - moves_left) if (13 - moves_left) > 0 else 0
            )
        else:
            avg_per_move = 0
        potential = avg_per_move * moves_left
        # scale
        return potential / 100.0

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Renders the current game state. If mode="human", prints to stdout.
        Return a string if needed for "ansi" or debugging.
        """
        dice_vals = self.state.current_dice
        dice_str = " ".join(str(d) if d > 0 else "-" for d in dice_vals)
        rolls = self.state.rolls_left

        lines = []
        lines.append(f"Dice: [{dice_str}] (rolls left: {rolls})")

        lines.append("\nScore sheet:")
        for cat in YahtzeeCategory:
            sc = self.state.score_sheet[cat]
            sc_str = str(sc) if sc is not None else "-"
            lines.append(f"{cat.name}: {sc_str}")

        bonus = self.calc_upper_bonus()
        if bonus > 0:
            lines.append(f"\nUpper Bonus: +{bonus}")

        text = "\n".join(lines)

        if mode == "human":
            print(text)
            return None
        else:
            return text

    def close(self):
        pass
