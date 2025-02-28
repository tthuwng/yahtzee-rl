import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from encoder import NUM_ACTIONS, ActionMapper, StateEncoder
from yahtzee_types import Action, ActionType, GameState, YahtzeeCategory


class RewardStrategy(Enum):
    """Different strategies for shaping the reward signal."""

    STANDARD = "standard"
    STRATEGIC = "strategic"
    NORMALIZED = "normalized"
    SPARSE = "sparse"
    POTENTIAL = "potential"
    ENHANCED = "enhanced"  # New enhanced strategy


class YahtzeeEnv(gym.Env):
    """Unified Yahtzee environment with gym interface."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        use_opponent_value: bool = False,
        reward_strategy: RewardStrategy = RewardStrategy.ENHANCED,  # Changed to enhanced
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.use_opponent_value = use_opponent_value
        self.reward_strategy = reward_strategy
        self.render_mode = render_mode

        self._last_potential: Optional[float] = None

        self.encoder = StateEncoder(use_opponent_value=use_opponent_value)
        self.action_mapper = ActionMapper()
        self._opponent_value = 0.5 if use_opponent_value else 0.0

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # Updated obs space to match enhanced state encoder
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.encoder.state_size,), dtype=np.float32
        )

        self.state: GameState = GameState(
            current_dice=np.zeros(5, dtype=int),
            rolls_left=3,
            score_sheet={cat: None for cat in YahtzeeCategory},
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> GameState:
        if seed is not None:
            super().reset(seed=seed)

        self._last_potential = None

        self.state = GameState(
            current_dice=np.zeros(5, dtype=int),
            rolls_left=3,
            score_sheet={cat: None for cat in YahtzeeCategory},
        )

        return self.state

    def step(self, action_idx: int) -> Tuple[GameState, float, bool, dict]:
        if action_idx < 0 or action_idx >= NUM_ACTIONS:
            raise ValueError(f"Invalid action index {action_idx}")

        action = self.action_mapper.index_to_action(action_idx)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action_type": action.kind.name}

        base_score = 0

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
            points = self.calc_score(category, self.state.current_dice)
            base_score = points
            shaped = self._shape_reward(
                float(points), action, done=False, base_score=points
            )
            reward = shaped

            self.state.score_sheet[category] = points
            info["category_scored"] = category.name
            info["points_scored"] = points

            # Check for game end
            if all(sc is not None for sc in self.state.score_sheet.values()):
                done = True
                bonus = self.calc_upper_bonus()
                reward += bonus
                info["upper_bonus"] = bonus

                # Calculate final score for info
                total_score = (
                    sum(
                        score
                        for score in self.state.score_sheet.values()
                        if score is not None
                    )
                    + bonus
                )
                info["final_score"] = total_score

                # Add bonus reward based on final score thresholds
                if total_score >= 300:
                    reward += 50.0  # Exceptional score
                elif total_score >= 250:
                    reward += 30.0  # Excellent score
                elif total_score >= 200:
                    reward += 10.0  # Good score

            self.state.current_dice = np.zeros(5, dtype=int)
            self.state.rolls_left = 3

        else:
            shaped = self._shape_reward(0.0, action, done=False, base_score=0)
            reward = shaped

        if done:
            reward = self._shape_reward(
                reward, action, done=True, base_score=base_score
            )

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
        return self.action_mapper.get_valid_actions(self.state)

    def calc_score(self, category: YahtzeeCategory, dice: np.ndarray) -> int:
        if not np.any(dice > 0):
            return 0

        counts = np.bincount(dice, minlength=7)[1:]
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
            has_three = any(c == 3 for c in counts)
            has_two = any(c == 2 for c in counts)
            return 25 if has_three and has_two else 0
        elif category == YahtzeeCategory.SMALL_STRAIGHT:
            sorted_unique = np.unique(dice)
            straights = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
            for st in straights:
                if all(x in sorted_unique for x in st):
                    return 30
            return 0
        elif category == YahtzeeCategory.LARGE_STRAIGHT:
            sorted_unique = np.unique(dice)
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

    def _shape_reward(
        self, reward: float, action: Action, done: bool, base_score: float
    ) -> float:
        if self.reward_strategy == RewardStrategy.STANDARD:
            return reward

        elif self.reward_strategy == RewardStrategy.STRATEGIC:
            if action.kind == ActionType.SCORE:
                return self.calc_strategic_reward(action.data, base_score)
            else:
                return reward

        elif self.reward_strategy == RewardStrategy.NORMALIZED:
            return float(np.clip(reward / 50.0, -1.0, 1.0))

        elif self.reward_strategy == RewardStrategy.SPARSE:
            return reward if done else 0.0

        elif self.reward_strategy == RewardStrategy.POTENTIAL:
            current_potential = self._compute_potential()
            if self._last_potential is None:
                shaped_reward = reward
            else:
                shaped_reward = reward + (
                    0.99 * current_potential - self._last_potential
                )
            self._last_potential = current_potential
            return shaped_reward

        elif self.reward_strategy == RewardStrategy.ENHANCED:
            if action.kind == ActionType.SCORE:
                return self.calc_enhanced_reward(action.data, base_score, done)
            else:
                # Weak reward signal for intelligent holding actions
                if action.kind == ActionType.HOLD:
                    # Calculate hold quality
                    hold_reward = self._evaluate_hold_quality(action.data)
                    return hold_reward
                return 0.0

        return reward

    def _evaluate_hold_quality(self, hold_mask: tuple) -> float:
        """Evaluate how good a hold action is."""
        if not self.state.current_dice.any():
            return 0.0

        dice = self.state.current_dice
        # Convert tuple to array
        hold_array = np.array(hold_mask, dtype=bool)
        held_dice = dice[hold_array]

        # If not holding anything, no extra reward
        if len(held_dice) == 0:
            return 0.0

        # Count values to detect patterns
        all_counts = np.bincount(dice, minlength=7)[1:]
        held_counts = np.bincount(held_dice, minlength=7)[1:]

        # Reward holding matching dice (higher for more matches)
        max_held = max(held_counts) if held_counts.any() else 0
        if max_held >= 3:
            return 0.5  # Good hold
        elif max_held == 2:
            return 0.2  # Decent hold

        # Reward holding dice for straights
        unique_vals = set(held_dice)
        if len(unique_vals) >= 3 and all(
            abs(a - b) <= 2 for a in unique_vals for b in unique_vals
        ):
            return 0.3  # Good for straight

        return 0.0  # No special reward

    def calc_strategic_reward(
        self, category: YahtzeeCategory, base_score: float
    ) -> float:
        """
        Modified to increase positive rewards and reduce harsh penalties for zero.
        """
        dice = self.state.current_dice
        counts = np.bincount(dice)[1:] if any(dice) else []
        max_count = max(counts) if len(counts) > 0 else 0

        bonus_reward = 0.0

        # Increase bonus for non-zero scoring
        if base_score > 0:
            # was +3.0 => now +5.0
            bonus_reward += 5.0
        else:
            # was -10.0 => now -4.0
            bonus_reward -= 4.0

        # Reward for sets
        if max_count >= 4 and (
            self.state.score_sheet.get(YahtzeeCategory.YAHTZEE) is None
        ):
            # was +12.0 => now +15.0
            bonus_reward += 15.0
        elif max_count >= 3:
            # was +5.0 => now +8.0
            bonus_reward += 8.0

        # Check if category is an upper category
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
            val_index = upper_cats.index(category)
            face_val = val_index + 1
            if base_score >= face_val * 3:
                # was +8.0 => now +10.0
                bonus_reward += 10.0
            else:
                bonus_reward += 2.0
            # scaled up from 0.5 => 0.8
            bonus_reward += face_val * 0.8

            if (upper_score_so_far + base_score) >= 63 and upper_filled < 5:
                # was +7.0 => now +10.0
                bonus_reward += 10.0

        # Special categories
        if category == YahtzeeCategory.FULL_HOUSE and base_score == 25:
            # was +7.0 => now +10.0
            bonus_reward += 10.0
        elif category == YahtzeeCategory.SMALL_STRAIGHT and base_score == 30:
            # was +8.0 => now +12.0
            bonus_reward += 12.0
        elif category == YahtzeeCategory.LARGE_STRAIGHT and base_score == 40:
            # was +10.0 => now +15.0
            bonus_reward += 15.0
        elif category == YahtzeeCategory.YAHTZEE and base_score == 50:
            # was +20.0 => now +30.0
            bonus_reward += 30.0

        # If it's zero in big categories, reduce penalty
        if base_score == 0 and category in [
            YahtzeeCategory.FULL_HOUSE,
            YahtzeeCategory.LARGE_STRAIGHT,
            YahtzeeCategory.YAHTZEE,
        ]:
            # was -7.0 => now -3.0
            bonus_reward -= 3.0

        return base_score + bonus_reward

    def calc_enhanced_reward(
        self, category: YahtzeeCategory, base_score: float, done: bool
    ) -> float:
        """
        Enhanced reward function with more sophisticated strategic guidance.
        """
        dice = self.state.current_dice
        counts = np.bincount(dice)[1:] if any(dice) else []
        max_count = max(counts) if len(counts) > 0 else 0

        bonus_reward = 0.0

        # === Base scoring rewards ===
        if base_score > 0:
            bonus_reward += 6.0  # Increased base reward for scoring
        else:
            bonus_reward -= 3.0  # Reduced penalty for zero

        # === Turn assessment ===
        # Factor in what turn we're in (how many categories filled)
        filled_categories = sum(
            1 for sc in self.state.score_sheet.values() if sc is not None
        )
        early_game = filled_categories < 4
        mid_game = 4 <= filled_categories < 8
        late_game = filled_categories >= 8

        # === Upper section strategy ===
        upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]

        # Calculate upper section stats
        upper_score_so_far = sum(self.state.score_sheet[cat] or 0 for cat in upper_cats)
        upper_filled = sum(
            1 for cat in upper_cats if self.state.score_sheet[cat] is not None
        )

        # Check bonus status
        upper_remaining = 6 - upper_filled
        if upper_remaining > 0:
            points_needed = max(0, 63 - upper_score_so_far)
            avg_needed_per_slot = points_needed / upper_remaining

            # Bonus prospects
            bonus_likely = avg_needed_per_slot <= 3.5
            bonus_possible = avg_needed_per_slot <= 5.0
            bonus_unlikely = avg_needed_per_slot > 5.0
        else:
            bonus_likely = False
            bonus_possible = False
            bonus_unlikely = True

        # Upper section scoring logic
        if category in upper_cats:
            val_index = upper_cats.index(category)
            face_val = val_index + 1

            # Base reward for better use of each category
            if base_score >= face_val * 3:
                bonus_reward += 12.0  # Increased from strategic reward
            elif base_score >= face_val * 2:
                bonus_reward += 5.0

            # Sliding scale for higher values
            bonus_reward += face_val * 1.0  # More reward for higher numbers

            # Hitting bonus threshold
            if (upper_score_so_far + base_score) >= 63 and not (
                upper_score_so_far >= 63
            ):
                bonus_reward += 25.0  # Big bonus for crossing the threshold

            # Strategic scoring based on game phase
            if early_game:
                if face_val <= 3 and base_score < face_val * 3:
                    bonus_reward -= 2.0  # Penalize poor low numbers early
            elif mid_game:
                if bonus_possible and bonus_unlikely and base_score < face_val * 3:
                    bonus_reward -= 5.0  # Losing bonus in mid-game
            elif late_game and bonus_likely:
                if (upper_score_so_far + base_score) < 63:
                    bonus_reward -= 8.0  # Losing guaranteed bonus in late game

        # === Special combinations ===
        # Full House
        if category == YahtzeeCategory.FULL_HOUSE:
            if base_score == 25:
                bonus_reward += 12.0
            elif max_count == 5:  # Using Yahtzee as Full House
                bonus_reward += 5.0
            elif base_score == 0 and not late_game:
                bonus_reward -= 5.0  # Penalize zero in early/mid-game

        # Small Straight
        elif category == YahtzeeCategory.SMALL_STRAIGHT:
            if base_score == 30:
                bonus_reward += 15.0
            elif base_score == 0 and not late_game:
                bonus_reward -= 5.0

        # Large Straight
        elif category == YahtzeeCategory.LARGE_STRAIGHT:
            if base_score == 40:
                bonus_reward += 18.0
            elif base_score == 0 and not late_game:
                bonus_reward -= 6.0

        # Yahtzee - highest value combination
        elif category == YahtzeeCategory.YAHTZEE:
            if base_score == 50:
                bonus_reward += 35.0  # Huge bonus
            elif base_score == 0:
                # Less penalty for zeroing Yahtzee in late game
                if late_game:
                    bonus_reward -= 2.0
                else:
                    bonus_reward -= 5.0

        # Three and Four of a Kind
        elif category == YahtzeeCategory.THREE_OF_A_KIND:
            if base_score >= 20:
                bonus_reward += 8.0
            elif base_score == 0 and not late_game:
                bonus_reward -= 4.0
        elif category == YahtzeeCategory.FOUR_OF_A_KIND:
            if base_score >= 25:
                bonus_reward += 12.0
            elif base_score == 0 and not late_game:
                bonus_reward -= 5.0

        # Chance - should maximize points
        elif category == YahtzeeCategory.CHANCE:
            if base_score >= 24:  # High Chance
                bonus_reward += 10.0
            elif base_score >= 20:  # Good Chance
                bonus_reward += 5.0
            elif base_score < 15 and not late_game:  # Poor use of Chance
                bonus_reward -= 3.0

        # === Game completion incentives ===
        if done:
            # Calculate final score excluding this bonus reward
            final_score = sum(
                score for score in self.state.score_sheet.values() if score is not None
            )
            final_score += self.calc_upper_bonus()

            # Reward thresholds
            if final_score >= 300:
                bonus_reward += 60.0  # Extraordinary score
            elif final_score >= 280:
                bonus_reward += 40.0  # Excellent score
            elif final_score >= 250:
                bonus_reward += 25.0  # Very good score
            elif final_score >= 220:
                bonus_reward += 15.0  # Good score
            elif final_score < 150:
                bonus_reward -= 10.0  # Poor score

        return base_score + bonus_reward

    def _compute_potential(self) -> float:
        total_score = sum(
            sc for sc in self.state.score_sheet.values() if sc is not None
        )
        moves_left = sum(1 for sc in self.state.score_sheet.values() if sc is None)
        if moves_left < 13:
            avg_per_move = (
                total_score / (13 - moves_left) if (13 - moves_left) > 0 else 0
            )
        else:
            avg_per_move = 0
        potential = avg_per_move * moves_left
        return potential / 100.0

    def render(self, mode: str = "human") -> Optional[str]:
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
