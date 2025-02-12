from typing import List

import numpy as np

from env import Action, GameState


class StateEncoder:
    """Efficient state encoder with focused feature set."""

    def __init__(self, use_opponent_value: bool = True) -> None:
        self.use_opponent_value = use_opponent_value

        # Calculate state size based on feature set
        self.state_size = 70

    def encode(self, state: GameState) -> np.ndarray:
        # Encode dice values (5 dice, one-hot encoded 1-6)
        dice_encoding = np.zeros(30)  # 5 dice * 6 possible values
        for i, value in enumerate(state.current_dice):
            if value > 0:
                dice_encoding[i * 6 + (value - 1)] = 1

        # Encode rolls left (one-hot encoded 0-3)
        rolls_left_encoding = np.zeros(3)
        if state.rolls_left > 0:
            rolls_left_encoding[state.rolls_left - 1] = 1

        # Encode score sheet (13 categories * 2 for filled and value)
        score_sheet_encoding = np.zeros(26)
        for i, (cat, score) in enumerate(state.score_sheet.items()):
            score_sheet_encoding[i * 2] = score is not None
            score_sheet_encoding[i * 2 + 1] = (score or 0) / 50.0  # Normalize values

        # Encode opponent score sheet if enabled
        opponent_encoding = np.zeros(11) if self.use_opponent_value else np.array([])
        if self.use_opponent_value and hasattr(state, "opponent_score_sheet"):
            total = sum(state.score_sheet[cat] or 0 for cat in state.score_sheet)
            opponent_encoding[0] = total / 100.0  # Normalize total
            for i, (cat, score) in enumerate(state.score_sheet.items()):
                opponent_encoding[i + 1] = (score or 0) / 50.0  # Normalize values

        return np.concatenate(
            [
                dice_encoding,  # 30 features
                rolls_left_encoding,  # 3 features
                score_sheet_encoding,  # 26 features
                opponent_encoding,  # 11 features if enabled
            ]
        )


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
