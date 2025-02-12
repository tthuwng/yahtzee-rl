from typing import Dict, List, Tuple

import numpy as np

from env import Action, YahtzeeCategory


class StateEncoder:
    """Efficient state encoder with focused feature set."""

    def __init__(self, use_opponent_value: bool = False) -> None:
        self.categories = list(YahtzeeCategory)
        self.num_categories = len(self.categories)
        self.use_opponent_value = use_opponent_value
        
        # Pre-compute category indices for faster lookup
        self.upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]
        
        # Calculate state size based on feature set
        self.state_size = (
            1  # Number of rerolls left
            + 6  # Count of each die value (1-6)
            + self.num_categories  # Player scorecard (filled categories)
            + 2  # Player upper and lower scores
            + (1 if use_opponent_value else 0)  # Opponent value (optional)
        )
        
        # Validate state size matches expected dimensions
        expected_size = 22 if not use_opponent_value else 23
        if self.state_size != expected_size:
            raise ValueError(
                f"State size mismatch. Expected {expected_size}, got {self.state_size}"
            )

    def _get_dice_counts(self, dice: np.ndarray) -> np.ndarray:
        """Get counts of each dice value (1-6)."""
        return np.bincount(dice, minlength=7)[1:]  # Skip index 0

    def _get_score_summary(self, scores: Dict[YahtzeeCategory, int]) -> Tuple[float, float]:
        """Calculate normalized upper and lower section scores."""
        upper_score = sum(scores[cat] or 0 for cat in self.upper_cats)
        lower_score = sum(
            scores[cat] or 0 
            for cat in self.categories 
            if cat not in self.upper_cats
        )
        
        # Normalize scores
        upper_score = min(upper_score / 63.0, 1.0)  # 63 is bonus threshold
        lower_score = min(lower_score / 200.0, 1.0)  # 200 is approximate max lower score
        
        return upper_score, lower_score

    def encode(self, state, opponent_value: float = 0.0) -> np.ndarray:
        """Convert game state to vector representation."""
        vec = np.zeros(self.state_size, dtype=np.float32)
        idx = 0

        # 1. Rolls left (normalized)
        vec[idx] = state.rolls_left / 3.0
        idx += 1

        # 2. Dice counts
        dice_counts = self._get_dice_counts(state.current_dice)
        counts_slice = slice(idx, idx + 6)
        vec[counts_slice] = dice_counts / 5.0  # Normalize by max possible count
        idx += 6

        # 3. Category flags (filled/unfilled)
        for cat in self.categories:
            vec[idx] = 1.0 if state.score_sheet[cat] is not None else 0.0
            idx += 1

        # 4. Upper and lower scores
        upper_score, lower_score = self._get_score_summary(state.score_sheet)
        vec[idx] = upper_score
        vec[idx + 1] = lower_score
        idx += 2

        # 5. Opponent value (if used)
        if self.use_opponent_value:
            vec[idx] = opponent_value

        return vec


class ActionMapper:
    """Maps between actions and indices."""

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
        """Create binary mask for valid actions."""
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[valid_actions] = 1.0
        return mask
