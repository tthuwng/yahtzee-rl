from typing import List

import numpy as np

from env import Action, YahtzeeCategory


class StateEncoder:
    """
    encodes the game state into a float vector for the neural net:
    - current dice (5 values from 0..6 => normalize /6.0)
    - rolls_left (1 float, 0..3 => /3)
    - which categories are filled? (13 binary flags)
    - normalized scores in each category (13 values, /50.0)
    Total size = 5 + 1 + 13 + 13 = 32
    """

    def __init__(self):
        self.state_size = 32
        self.categories = list(YahtzeeCategory)

    def encode(self, state) -> np.ndarray:
        vec = np.zeros(self.state_size, dtype=np.float32)

        # encode dice values (normalized by 6)
        vec[0:5] = state.current_dice / 6.0

        # encode rolls left (normalized by 3)
        vec[5] = state.rolls_left / 3.0

        # encode which categories are filled and their scores
        offset_filled = 6
        offset_scores = offset_filled + 13
        for i, cat in enumerate(self.categories):
            score = state.score_sheet[cat]
            if score is not None:
                vec[offset_filled + i] = 1.0
                # normalize score by 50 (max possible in any category)
                vec[offset_scores + i] = min(score, 50) / 50.0

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
