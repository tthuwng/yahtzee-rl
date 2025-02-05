from typing import List, Dict, Tuple

import numpy as np

from env import Action, YahtzeeCategory


class StateEncoder:
    """Encodes game state into a vector with enhanced features for better learning."""

    def __init__(self) -> None:
        self.categories = list(YahtzeeCategory)
        # Expanded state size to include more informative features
        self.state_size = (
            5 +      # Current dice values
            1 +      # Rolls left
            13 +     # Category filled flags
            13 +     # Category scores
            6 +      # Dice value counts
            1 +      # Upper section total
            1 +      # Upper bonus eligibility
            6 +      # Potential upper section scores
            5        # Potential combination scores (3K, 4K, FH, SS, LS)
        )           # Total: 51 features

    def _get_dice_counts(self, dice: np.ndarray) -> np.ndarray:
        """Get normalized counts of each dice value."""
        counts = np.zeros(6)
        for i in range(6):
            counts[i] = np.sum(dice == (i + 1))
        return counts / 5.0  # Normalize by max possible count

    def _get_upper_section_total(self, scores: Dict[YahtzeeCategory, int]) -> float:
        """Calculate normalized upper section total."""
        upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES
        ]
        total = sum(scores[cat] or 0 for cat in upper_cats)
        return min(total / 63.0, 1.0)  # Normalize by bonus threshold

    def _get_potential_scores(self, dice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate potential scores for unfilled categories."""
        # Upper section potential scores
        upper_scores = np.zeros(6)
        for i in range(6):
            count = np.sum(dice == (i + 1))
            upper_scores[i] = count * (i + 1) / 30.0  # Normalize by max possible (5 * 6 = 30)
        
        # Combination scores
        combo_scores = np.zeros(5)
        if any(dice):  # Only calculate if dice are showing
            counts = np.bincount(dice)[1:] if any(dice) else []
            dice_sum = dice.sum()
            
            # Three of a Kind
            combo_scores[0] = dice_sum / 30.0 if any(c >= 3 for c in counts) else 0
            
            # Four of a Kind
            combo_scores[1] = dice_sum / 30.0 if any(c >= 4 for c in counts) else 0
            
            # Full House
            combo_scores[2] = 25.0 / 25.0 if any(c == 3 for c in counts) and any(c == 2 for c in counts) else 0
            
            # Small Straight
            sorted_unique = np.unique(dice)
            for straight in [[1,2,3,4], [2,3,4,5], [3,4,5,6]]:
                if all(x in sorted_unique for x in straight):
                    combo_scores[3] = 30.0 / 30.0
                    break
            
            # Large Straight
            if len(sorted_unique) == 5 and (
                all(x in sorted_unique for x in [1,2,3,4,5]) or
                all(x in sorted_unique for x in [2,3,4,5,6])
            ):
                combo_scores[4] = 40.0 / 40.0
        
        return upper_scores, combo_scores

    def encode(self, state) -> np.ndarray:
        """Convert game state to enhanced vector representation."""
        vec = np.zeros(self.state_size, dtype=np.float32)
        idx = 0

        # Dice values (normalized)
        vec[idx:idx+5] = state.current_dice / 6.0
        idx += 5

        # Rolls left (normalized)
        vec[idx] = state.rolls_left / 3.0
        idx += 1

        # Category flags and scores
        for cat in self.categories:
            score = state.score_sheet[cat]
            vec[idx] = 1.0 if score is not None else 0.0  # Category filled flag
            vec[idx + 13] = min(score / 50.0, 1.0) if score is not None else 0.0  # Normalized score
        idx += 26  # 13 flags + 13 scores

        # Dice value counts
        vec[idx:idx+6] = self._get_dice_counts(state.current_dice)
        idx += 6

        # Upper section progress
        vec[idx] = self._get_upper_section_total(state.score_sheet)
        idx += 1

        # Upper bonus eligibility
        upper_total = sum(
            state.score_sheet[cat] or 0
            for cat in self.categories[:6]  # Upper section categories
        )
        vec[idx] = 1.0 if upper_total >= 63 else upper_total / 63.0
        idx += 1

        # Potential scores for unfilled categories
        upper_scores, combo_scores = self._get_potential_scores(state.current_dice)
        vec[idx:idx+6] = upper_scores
        idx += 6
        vec[idx:idx+5] = combo_scores

        return vec


class ActionMapper:
    """maps between action indices and Action objects."""

    def __init__(self):
        from env import ACTION_TO_IDX, ALL_ACTIONS, IDX_TO_ACTION

        self.actions = ALL_ACTIONS
        self.action_size = len(ALL_ACTIONS)
        self.action_to_idx = ACTION_TO_IDX
        self.idx_to_action = IDX_TO_ACTION

    def action_to_index(self, action: Action) -> int:
        return self.action_to_idx[action]

    def index_to_action(self, index: int) -> Action:
        return self.idx_to_action[index]

    def valid_action_mask(self, valid_actions: List[int]) -> np.ndarray:
        """create a binary mask for valid actions."""
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[valid_actions] = 1.0
        return mask
