from typing import Dict, List, Optional, Tuple

import numpy as np

from yahtzee_types import (
    Action,
    ActionType,
    GameState,
    YahtzeeCategory,
)

# Action mapping constants
NUM_ACTIONS = 46  # 1 ROLL + 32 HOLD + 13 SCORE
ALL_ACTIONS = []  # Will be populated in ActionMapper
ACTION_TO_IDX = {}  # Will be populated in ActionMapper
IDX_TO_ACTION = {}  # Will be populated in ActionMapper


class StateEncoder:
    """Basic state encoder for Yahtzee."""

    def __init__(self, use_opponent_value: bool = False) -> None:
        self.categories = list(YahtzeeCategory)
        self.num_categories = len(self.categories)
        self.use_opponent_value = use_opponent_value

        # Basic state features
        self.state_size = (
            1  # Number of rerolls left
            + 6  # Count of each die value (1-6)
            + self.num_categories  # Player scorecard (filled categories)
            + 2  # Player upper and lower scores
            + (
                1 if use_opponent_value else 0
            )  # Opponent value (optional)
        )

    def _get_dice_counts(self, dice: np.ndarray) -> np.ndarray:
        """Get counts of each dice value (1-6)."""
        return np.bincount(dice, minlength=7)[1:]  # Skip index 0

    def _get_score_summary(
        self, scores: Dict[YahtzeeCategory, Optional[int]]
    ) -> Tuple[float, float]:
        """Calculate normalized upper and lower section scores."""
        upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]

        upper_score = sum(scores[cat] or 0 for cat in upper_cats)
        lower_score = sum(
            scores[cat] or 0
            for cat in self.categories
            if cat not in upper_cats
        )

        # Normalize scores
        upper_score_norm = min(
            upper_score / 63.0, 1.0
        )  # 63 is bonus threshold
        lower_score_norm = min(
            lower_score / 200.0, 1.0
        )  # 200 is approximate max lower score

        return upper_score_norm, lower_score_norm

    def encode(
        self, state: GameState, opponent_value: float = 0.0
    ) -> np.ndarray:
        """Convert game state to vector representation."""
        vec = np.zeros(self.state_size, dtype=np.float32)
        idx = 0

        # 1. Rolls left (normalized)
        vec[idx] = state.rolls_left / 3.0
        idx += 1

        # 2. Dice counts
        dice_counts = self._get_dice_counts(state.current_dice)
        vec[idx : idx + 6] = (
            dice_counts / 5.0
        )  # Normalize by max possible count
        idx += 6

        # 3. Category flags (filled/unfilled)
        for cat in self.categories:
            vec[idx] = (
                1.0 if state.score_sheet[cat] is not None else 0.0
            )
            idx += 1

        # 4. Upper and lower scores
        upper_score, lower_score = self._get_score_summary(
            state.score_sheet
        )
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
        # Generate all possible actions
        self.hold_masks = []
        for bits in range(32):
            mask = [(bits & (1 << i)) != 0 for i in range(5)]
            self.hold_masks.append(np.array(mask, dtype=bool))

        # Clear global mappings
        global ALL_ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION
        ALL_ACTIONS.clear()
        ACTION_TO_IDX.clear()
        IDX_TO_ACTION.clear()

        # ROLL action
        ALL_ACTIONS.append(Action(ActionType.ROLL))

        # HOLD actions
        for mask in self.hold_masks:
            hold_action = Action(ActionType.HOLD, tuple(mask))
            ALL_ACTIONS.append(hold_action)

        # SCORE actions
        for cat in YahtzeeCategory:
            ALL_ACTIONS.append(Action(ActionType.SCORE, cat))

        # Build action maps
        for i, act in enumerate(ALL_ACTIONS):
            ACTION_TO_IDX[act] = i
            IDX_TO_ACTION[i] = act

        self.action_size = len(ALL_ACTIONS)

    def action_to_index(self, action: Action) -> int:
        """Convert Action to index."""
        return ACTION_TO_IDX[action]

    def index_to_action(self, index: int) -> Action:
        """Convert index to Action."""
        return IDX_TO_ACTION[index]

    def get_valid_actions(self, state: GameState) -> List[int]:
        """Get list of valid action indices for current state."""
        valid = []
        has_rolls = state.rolls_left > 0
        not_finished = None in state.score_sheet.values()

        if has_rolls and not_finished:
            # If showing no dice, can only ROLL
            if not np.any(state.current_dice):
                roll_idx = ACTION_TO_IDX[Action(ActionType.ROLL)]
                valid.append(roll_idx)
            else:
                # can do ROLL or HOLD combos
                roll_idx = ACTION_TO_IDX[Action(ActionType.ROLL)]
                valid.append(roll_idx)
                for mask in self.hold_masks:
                    if np.any(mask):
                        hold_act = Action(
                            ActionType.HOLD, tuple(mask)
                        )
                        valid.append(ACTION_TO_IDX[hold_act])

        # can score if dice are showing
        if np.any(state.current_dice):
            for cat in YahtzeeCategory:
                if state.score_sheet[cat] is None:
                    score_act = Action(ActionType.SCORE, cat)
                    valid.append(ACTION_TO_IDX[score_act])

        return valid
