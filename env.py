from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class YahtzeeCategory(Enum):
    ONES = auto()
    TWOS = auto()
    THREES = auto()
    FOURS = auto()
    FIVES = auto()
    SIXES = auto()
    THREE_OF_A_KIND = auto()
    FOUR_OF_A_KIND = auto()
    FULL_HOUSE = auto()
    SMALL_STRAIGHT = auto()
    LARGE_STRAIGHT = auto()
    YAHTZEE = auto()
    CHANCE = auto()


def all_hold_masks():
    """return list of all possible 5-dice hold combinations (32 total)."""
    masks = []
    for bits in range(32):  # from 0 to 31
        mask = [(bits & (1 << i)) != 0 for i in range(5)]
        masks.append(np.array(mask, dtype=bool))
    return masks


class ActionType(Enum):
    ROLL = auto()
    HOLD_MASK = auto()
    SCORE = auto()


class Action:
    """
    a composite action that could be:
      - (ROLL, None)
      - (HOLD_MASK, np.ndarray of shape(5,) bool)
      - (SCORE, YahtzeeCategory)
    """

    def __init__(self, kind: ActionType, data=None):
        self.kind = kind
        self.data = data  # either a boolean mask (5,) or a YahtzeeCategory

    def __repr__(self):
        return f"Action({self.kind}, {self.data})"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        if self.kind != other.kind:
            return False
        if isinstance(self.data, np.ndarray):
            return np.array_equal(self.data, other.data)
        return self.data == other.data

    def __hash__(self):
        if isinstance(self.data, np.ndarray):
            return hash((self.kind, tuple(self.data)))
        return hash((self.kind, self.data))


# pre-generate all possible actions
ALL_ACTIONS = []

# 1) ROLL:
ALL_ACTIONS.append(Action(ActionType.ROLL, None))

# 2) 32 hold masks:
HOLD_MASKS = all_hold_masks()
for mask in HOLD_MASKS:
    ALL_ACTIONS.append(Action(ActionType.HOLD_MASK, mask))

# 3) 13 scoring categories:
for cat in YahtzeeCategory:
    ALL_ACTIONS.append(Action(ActionType.SCORE, cat))

NUM_ACTIONS = len(ALL_ACTIONS)  # 1 + 32 + 13 = 46
IDX_TO_ACTION = {i: act for i, act in enumerate(ALL_ACTIONS)}
ACTION_TO_IDX = {act: i for i, act in enumerate(ALL_ACTIONS)}


@dataclass
class GameState:
    """represents the current state of a Yahtzee game"""

    current_dice: np.ndarray  # array of shape (5,) with values 1..6
    rolls_left: int  # integer, how many rolls left this turn
    score_sheet: Dict[YahtzeeCategory, Optional[int]]  # category -> int or None
    total_points: int = 0  # Track total points internally


class YahtzeeEnv:
    def __init__(self):
        self.reset()

    def reset(self) -> GameState:
        """
        start a new game with empty scoresheet, 3 rolls available, no dice yet.
        """
        self.state = GameState(
            current_dice=np.zeros(5, dtype=int),
            rolls_left=3,
            score_sheet={cat: None for cat in YahtzeeCategory},
            total_points=0,
        )
        return self.state

    def roll_dice(self, hold_mask: np.ndarray):
        """roll only the dice that are not held in the mask."""
        if self.state.rolls_left <= 0:
            raise ValueError("Cannot roll with no rolls left!")
        for i in range(5):
            if not hold_mask[i]:  # not held => roll it
                self.state.current_dice[i] = np.random.randint(1, 7)
        self.state.rolls_left -= 1

    def calc_upper_score(self) -> int:
        """sum of ONES..SIXES that have been scored so far (ignoring None)."""
        s = 0
        upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]
        for cat in upper_cats:
            if self.state.score_sheet[cat] is not None:
                s += self.state.score_sheet[cat]
        return s

    def calc_category_score(self, category: YahtzeeCategory, dice: np.ndarray) -> int:
        """
        calculate score for a given category with proper upper section scoring.
        """
        # Count occurrences
        counts = np.bincount(dice, minlength=7)  # index 0 unused
        unique_vals = np.unique(dice)

        if category == YahtzeeCategory.ONES:
            return counts[1] * 1
        elif category == YahtzeeCategory.TWOS:
            return counts[2] * 2
        elif category == YahtzeeCategory.THREES:
            return counts[3] * 3
        elif category == YahtzeeCategory.FOURS:
            return counts[4] * 4
        elif category == YahtzeeCategory.FIVES:
            return counts[5] * 5
        elif category == YahtzeeCategory.SIXES:
            return counts[6] * 6
        elif category == YahtzeeCategory.THREE_OF_A_KIND:
            if np.any(counts >= 3):
                return dice.sum()
            return 0
        elif category == YahtzeeCategory.FOUR_OF_A_KIND:
            if np.any(counts >= 4):
                return dice.sum()
            return 0
        elif category == YahtzeeCategory.FULL_HOUSE:
            if len(unique_vals) == 2 and (3 in counts and 2 in counts):
                return 25
            return 0
        elif category == YahtzeeCategory.SMALL_STRAIGHT:
            for seq in [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]:
                if all(v in unique_vals for v in seq):
                    return 30
            return 0
        elif category == YahtzeeCategory.LARGE_STRAIGHT:
            if len(unique_vals) == 5 and (
                all(v in unique_vals for v in [1, 2, 3, 4, 5])
                or all(v in unique_vals for v in [2, 3, 4, 5, 6])
            ):
                return 40
            return 0
        elif category == YahtzeeCategory.YAHTZEE:
            if len(unique_vals) == 1:
                return 50
            return 0
        elif category == YahtzeeCategory.CHANCE:
            return dice.sum()
        else:
            raise ValueError(f"Unknown category: {category}")

    def get_valid_actions(self) -> List[int]:
        """return list of valid action indices in current state."""
        valid = []
        # if we can still roll, ROLL is valid:
        if self.state.rolls_left > 0:
            valid.append(ACTION_TO_IDX[Action(ActionType.ROLL, None)])
            # all hold combos are valid
            for mask in HOLD_MASKS:
                valid.append(ACTION_TO_IDX[Action(ActionType.HOLD_MASK, mask)])
        # for each unfilled category, the "SCORE" action is valid
        for cat in YahtzeeCategory:
            if self.state.score_sheet[cat] is None:
                valid.append(ACTION_TO_IDX[Action(ActionType.SCORE, cat)])
        return valid

    def _calc_pattern_reward(self, dice: np.ndarray) -> float:
        """Calculate small rewards for promising dice patterns."""
        if len(dice) == 0 or np.all(dice == 0):
            return 0.0

        counts = np.bincount(dice, minlength=7)
        unique_vals = np.unique(dice[dice > 0])
        reward = 0.0

        # reward for collecting same numbers
        max_count = np.max(counts)
        if max_count >= 3:
            reward += 0.2  # Three of a kind
        if max_count >= 4:
            reward += 0.3  # Four of a kind
        if max_count == 5:
            reward += 0.5  # Yahtzee potential

        # reward for straights progress
        sorted_vals = np.sort(unique_vals)
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] == sorted_vals[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        if max_consecutive >= 4:
            reward += 0.3  # small straight potential
        if max_consecutive >= 5:
            reward += 0.2  # large straight potential

        # reward for full house potential
        if len(unique_vals) == 2 and (2 in counts and 3 in counts):
            reward += 0.4

        return reward

    def step(self, action_idx: int) -> Tuple[GameState, float, bool, dict]:
        """
        take one step given an action index.
        returns (next_state, reward, done, info).
        - If action is ROLL or HOLD_MASK, reward includes pattern rewards
        - If action is SCORE, returns immediate scoring reward
        - Adds upper section bonus at game end
        """
        action = IDX_TO_ACTION[action_idx]
        reward = 0.0
        done = False

        if action.kind == ActionType.ROLL:
            # roll with no dice held
            hold_mask = np.zeros(5, dtype=bool)
            self.roll_dice(hold_mask)
            reward = self._calc_pattern_reward(self.state.current_dice) * 0.2

        elif action.kind == ActionType.HOLD_MASK:
            # roll with the given hold mask
            old_dice = self.state.current_dice.copy()
            self.roll_dice(action.data)
            # reward for good hold patterns
            held_dice = old_dice[action.data]
            reward = self._calc_pattern_reward(held_dice) * 0.2

        elif action.kind == ActionType.SCORE:
            cat = action.data
            if self.state.score_sheet[cat] is not None:
                raise ValueError(f"Category {cat} already filled!")

            # calculate points and give immediate reward
            points = self.calc_category_score(cat, self.state.current_dice)
            self.state.score_sheet[cat] = points
            self.state.total_points += points
            reward = float(points)

            # check if game is done
            if all(v is not None for v in self.state.score_sheet.values()):
                # Add upper bonus if earned
                upper_total = self.calc_upper_score()
                if upper_total >= 63:
                    self.state.total_points += 35
                    reward += 35.0  # bonus to final reward
                done = True
            else:
                # New turn
                self.state.current_dice = np.zeros(5, dtype=int)
                self.state.rolls_left = 3

        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        return self.state, reward, done, {}

    def render(self) -> str:
        """render the current game state as a string."""
        lines = []
        dice_str = f"Dice: {self.state.current_dice}"
        rolls_str = f"(rolls left: {self.state.rolls_left})"
        lines.append(f"{dice_str} {rolls_str}")
        lines.append("\nScore sheet:")
        for cat in YahtzeeCategory:
            sc = self.state.score_sheet[cat]
            lines.append(f"  {cat.name}: {sc if sc is not None else '-'}")
        upper_score = self.calc_upper_score()
        lines.append(f"\nUpper section total: {upper_score}")
        if upper_score >= 63:
            lines.append("Upper bonus: +35")
        return "\n".join(lines)
