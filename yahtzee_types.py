from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np


class YahtzeeCategory(Enum):
    """Basic categories in Yahtzee."""

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


class ActionType(Enum):
    """Basic action types in Yahtzee."""

    ROLL = auto()  # Roll all dice
    HOLD = auto()  # Hold some dice
    SCORE = auto()  # Score a category


@dataclass(frozen=True)
class Action:
    """Action in Yahtzee game."""

    kind: ActionType
    data: Optional[object] = (
        None  # bool array for HOLD, YahtzeeCategory for SCORE
    )


@dataclass
class GameState:
    """Current state of Yahtzee game."""

    current_dice: np.ndarray  # Values of 5 dice
    rolls_left: int  # Rolls remaining this turn
    score_sheet: Dict[
        YahtzeeCategory, Optional[int]
    ]  # Category -> score or None
