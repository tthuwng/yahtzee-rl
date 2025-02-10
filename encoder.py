from typing import Dict, List, Tuple

import numpy as np

from env import Action, YahtzeeCategory


class StateEncoder:
    """Enhanced state encoder with richer features for better learning."""

    def __init__(self) -> None:
        self.categories = list(YahtzeeCategory)
        # Expanded state size with additional features
        self.state_size = (
            5  # Current dice values (normalized)
            + 1  # Rolls left (normalized)
            + 13  # Category filled flags
            + 13  # Category scores (normalized)
            + 6  # Dice value counts (normalized)
            + 1  # Upper section total (normalized)
            + 1  # Upper bonus eligibility
            + 6  # Potential upper section scores
            + 5  # Potential combination scores
            + 13  # One-hot encoding of last scored category
            + 5  # Dice position importance scores
            + 1  # Turn number (normalized)
        )  # Total: 70 features

        # Pre-compute category indices for faster lookup
        self.upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]

        # Cache for faster computation
        self.straight_patterns = {
            "small": [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
            "large": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
        }

    def _get_dice_counts(self, dice: np.ndarray) -> np.ndarray:
        """Get normalized counts of each dice value."""
        counts = np.bincount(dice, minlength=7)[1:]  # Skip index 0
        return counts / 5.0  # Normalize by max possible count

    def _get_upper_section_total(
        self,
        scores: Dict[YahtzeeCategory, int],
    ) -> float:
        """Calculate normalized upper section total."""
        total = sum(scores[cat] or 0 for cat in self.upper_cats)
        return min(total / 63.0, 1.0)  # Normalize by bonus threshold

    def _get_potential_scores(
        self,
        dice: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate potential scores with optimized computation."""
        # Upper section potential scores (vectorized)
        upper_scores = np.zeros(6, dtype=np.float32)
        if any(dice):
            counts = np.bincount(dice, minlength=7)[1:]
            upper_scores = (counts * np.arange(1, 7)) / 30.0

        # Combination scores (vectorized)
        combo_scores = np.zeros(5, dtype=np.float32)
        if any(dice):
            dice_sum = dice.sum()

            # Three/Four of a Kind
            max_count = counts.max()
            combo_scores[0] = dice_sum / 30.0 if max_count >= 3 else 0
            combo_scores[1] = dice_sum / 30.0 if max_count >= 4 else 0

            # Full House
            sorted_counts = np.sort(counts)
            has_three = sorted_counts[-1] == 3
            has_two = sorted_counts[-2] == 2
            combo_scores[2] = 1.0 if has_three and has_two else 0

            # Small/Large Straight
            sorted_unique = np.unique(dice)
            for straight in self.straight_patterns["small"]:
                if all(x in sorted_unique for x in straight):
                    combo_scores[3] = 1.0
                    break

            if len(sorted_unique) == 5:
                for straight in self.straight_patterns["large"]:
                    if all(x in sorted_unique for x in straight):
                        combo_scores[4] = 1.0
                        break

        return upper_scores, combo_scores

    def _get_dice_position_importance(self, dice: np.ndarray) -> np.ndarray:
        """Calculate importance scores for each dice position."""
        if not any(dice):
            return np.zeros(5, dtype=np.float32)

        counts = np.bincount(dice, minlength=7)[1:]
        max_count = counts.max()
        max_val = counts.argmax() + 1

        importance = np.zeros(5, dtype=np.float32)
        for i, val in enumerate(dice):
            if val == max_val:
                importance[i] = max_count / 5.0
            elif val > 0:
                importance[i] = 0.5  # Medium importance

        return importance

    def encode(self, state) -> np.ndarray:
        """Convert game state to enhanced vector representation."""
        vec = np.zeros(self.state_size, dtype=np.float32)
        idx = 0

        # Dice values (normalized)
        dice_slice = slice(idx, idx + 5)
        vec[dice_slice] = state.current_dice / 6.0
        idx += 5

        # Rolls left (normalized)
        vec[idx] = state.rolls_left / 3.0
        idx += 1

        # Category flags and normalized scores
        for cat in self.categories:
            score = state.score_sheet[cat]
            vec[idx] = 1.0 if score is not None else 0.0
            norm_score = min(score / 50.0, 1.0) if score is not None else 0.0
            vec[idx + 13] = norm_score
        idx += 26

        # Dice value counts
        counts_slice = slice(idx, idx + 6)
        vec[counts_slice] = self._get_dice_counts(state.current_dice)
        idx += 6

        # Upper section progress
        vec[idx] = self._get_upper_section_total(state.score_sheet)
        idx += 1

        # Upper bonus eligibility
        upper_total = sum(state.score_sheet[cat] or 0 for cat in self.upper_cats)
        vec[idx] = 1.0 if upper_total >= 63 else upper_total / 63.0
        idx += 1

        # Potential scores
        upper_scores, combo_scores = self._get_potential_scores(state.current_dice)
        upper_slice = slice(idx, idx + 6)
        vec[upper_slice] = upper_scores
        idx += 6
        combo_slice = slice(idx, idx + 5)
        vec[combo_slice] = combo_scores
        idx += 5

        # One-hot encoding of last scored category
        last_scored = None
        for cat in reversed(self.categories):
            if state.score_sheet[cat] is not None:
                last_scored = cat
                break
        if last_scored:
            vec[idx + self.categories.index(last_scored)] = 1.0
        idx += 13

        # Dice position importance
        imp_slice = slice(idx, idx + 5)
        vec[imp_slice] = self._get_dice_position_importance(state.current_dice)
        idx += 5

        # Turn number (normalized by max turns)
        filled = sum(1 for s in state.score_sheet.values() if s is not None)
        vec[idx] = filled / 13.0

        return vec


class ActionMapper:
    """Optimized action mapping with tensor operations."""

    def __init__(self):
        from env import ACTION_TO_IDX, ALL_ACTIONS, IDX_TO_ACTION

        self.actions = ALL_ACTIONS
        self.action_size = len(ALL_ACTIONS)
        self.action_to_idx = ACTION_TO_IDX
        self.idx_to_action = IDX_TO_ACTION

        # Pre-compute action masks for common patterns
        self.roll_mask = np.zeros(self.action_size, dtype=np.float32)
        self.roll_mask[0] = 1  # Assuming ROLL is first action

        self.hold_mask = np.zeros(self.action_size, dtype=np.float32)
        self.hold_mask[1:33] = 1  # Assuming HOLD actions are 1-32

        self.score_mask = np.zeros(self.action_size, dtype=np.float32)
        self.score_mask[33:] = 1  # Assuming SCORE actions are 33+

    def action_to_index(self, action: Action) -> int:
        return self.action_to_idx[action]

    def index_to_action(self, index: int) -> Action:
        return self.idx_to_action[index]

    def valid_action_mask(self, valid_actions: List[int]) -> np.ndarray:
        """Create an efficient binary mask for valid actions."""
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[valid_actions] = 1.0
        return mask
