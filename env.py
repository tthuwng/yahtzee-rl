from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "YahtzeeCategory",
    "ActionType",
    "Action",
    "GameState",
    "YahtzeeEnv",
    "NUM_ACTIONS",
    "IDX_TO_ACTION",
]

NUM_ACTIONS = 46  # 1 ROLL + 32 HOLD + 13 SCORE actions
IDX_TO_ACTION = {}

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

class ActionType(Enum):
    ROLL = auto()
    HOLD = auto()
    SCORE = auto()

@dataclass(frozen=True)
class Action:
    kind: ActionType
    data: Optional[object] = None

    def __hash__(self):
        if self.kind == ActionType.HOLD and self.data is not None:
            return hash((self.kind, tuple(map(bool, self.data))))
        return hash((self.kind, self.data))

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        if self.kind != other.kind:
            return False
        if (
            self.kind == ActionType.HOLD
            and self.data is not None
            and other.data is not None
        ):
            return np.array_equal(self.data, other.data)
        return self.data == other.data

@dataclass
class GameState:
    current_dice: np.ndarray
    rolls_left: int
    score_sheet: Dict[YahtzeeCategory, Optional[int]]

class YahtzeeEnv:
    def __init__(self):
        self.hold_masks = []
        for bits in range(32):
            mask = [(bits & (1 << i)) != 0 for i in range(5)]
            self.hold_masks.append(np.array(mask, dtype=bool))

        self.all_actions = []
        self.all_actions.append(Action(ActionType.ROLL))
        for mask in self.hold_masks:
            self.all_actions.append(Action(ActionType.HOLD, mask))
        for cat in YahtzeeCategory:
            self.all_actions.append(Action(ActionType.SCORE, cat))

        self.action_to_idx = {act: i for i, act in enumerate(self.all_actions)}
        self.idx_to_action = {i: act for i, act in enumerate(self.all_actions)}
        global IDX_TO_ACTION
        IDX_TO_ACTION.update(self.idx_to_action)
        self.num_actions = len(self.all_actions)

    def reset(self) -> GameState:
        self.state = GameState(
            current_dice=np.zeros(5, dtype=int),
            rolls_left=3,
            score_sheet={cat: None for cat in YahtzeeCategory},
        )
        return self.state

    def roll_dice(self, hold_mask: np.ndarray) -> None:
        if self.state.rolls_left <= 0:
            raise ValueError("No rolls left!")
        for i in range(5):
            if not hold_mask[i]:
                self.state.current_dice[i] = np.random.randint(1, 7)
        self.state.rolls_left -= 1

    def get_valid_actions(self) -> List[int]:
        valid = []
        has_rolls = self.state.rolls_left > 0
        not_finished = None in self.state.score_sheet.values()
        if has_rolls and not_finished:
            if not np.any(self.state.current_dice):
                valid.append(self.action_to_idx[Action(ActionType.ROLL)])
            else:
                valid.append(self.action_to_idx[Action(ActionType.ROLL)])
                for mask in self.hold_masks:
                    if np.any(mask):
                        valid.append(self.action_to_idx[Action(ActionType.HOLD, mask)])

        if np.any(self.state.current_dice):
            for cat in YahtzeeCategory:
                if self.state.score_sheet[cat] is None:
                    valid.append(self.action_to_idx[Action(ActionType.SCORE, cat)])
        return valid

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
            for val, c in enumerate(counts, 1):
                if c >= 3:
                    return dice.sum()
            return 0
        elif category == YahtzeeCategory.FOUR_OF_A_KIND:
            for val, c in enumerate(counts, 1):
                if c >= 4:
                    return dice.sum()
            return 0
        elif category == YahtzeeCategory.FULL_HOUSE:
            has_three = any(c == 3 for c in counts)
            has_two = any(c == 2 for c in counts)
            return 25 if (has_three and has_two) else 0
        elif category == YahtzeeCategory.SMALL_STRAIGHT:
            sorted_unique = np.unique(dice)
            for straight in [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]:
                if all(x in sorted_unique for x in straight):
                    return 30
            return 0
        elif category == YahtzeeCategory.LARGE_STRAIGHT:
            sorted_unique = np.unique(dice)
            if len(sorted_unique) == 5 and (
                all(x in sorted_unique for x in [1, 2, 3, 4, 5]) or
                all(x in sorted_unique for x in [2, 3, 4, 5, 6])
            ):
                return 40
            return 0
        elif category == YahtzeeCategory.YAHTZEE:
            return 50 if np.any(counts == 5) else 0
        elif category == YahtzeeCategory.CHANCE:
            return dice.sum()
        else:
            raise ValueError(f"Unknown category: {category}")

    def calc_upper_bonus(self) -> int:
        upper_score = sum(
            self.state.score_sheet[cat] or 0
            for cat in [
                YahtzeeCategory.ONES,
                YahtzeeCategory.TWOS,
                YahtzeeCategory.THREES,
                YahtzeeCategory.FOURS,
                YahtzeeCategory.FIVES,
                YahtzeeCategory.SIXES,
            ]
        )
        return 35 if upper_score >= 63 else 0

    def calc_strategic_reward(
        self, category: YahtzeeCategory, base_score: float
    ) -> float:
        """
        Reward shaping focusing on higher synergy for big combos and stronger push
        for achieving or surpassing 63 points in upper section as early as possible.
        """
        dice = self.state.current_dice
        counts = np.bincount(dice)[1:] if any(dice) else []
        max_count = max(counts) if any(counts) else 0
        # Track upper
        upper_cats = [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]
        upper_score_so_far = sum(self.state.score_sheet[cat] or 0 for cat in upper_cats)
        # Baseline is actual points
        bonus_reward = 0.0

        # baseline +2 for any scoring
        bonus_reward += 2.0

        # reward big combos
        if max_count >= 4:
            bonus_reward += 7.0
        elif max_count == 3:
            bonus_reward += 3.0

        # upper synergy
        if category in upper_cats:
            val_index = upper_cats.index(category)
            val = val_index + 1
            # encourage bigger chunk in upper
            if base_score >= val * 3:
                bonus_reward += 5.0
            else:
                bonus_reward += 1.5

            # if we pass 63 or are close, bigger push
            if (upper_score_so_far + base_score) >= 63:
                bonus_reward += 8.0

        # bigger combos
        if category == YahtzeeCategory.FULL_HOUSE and base_score == 25:
            bonus_reward += 4.0
        elif category == YahtzeeCategory.SMALL_STRAIGHT and base_score == 30:
            bonus_reward += 5.0
        elif category == YahtzeeCategory.LARGE_STRAIGHT and base_score == 40:
            bonus_reward += 6.0
        elif category == YahtzeeCategory.YAHTZEE and base_score == 50:
            bonus_reward += 12.0

        # penalize zero
        if base_score == 0:
            if category == YahtzeeCategory.CHANCE:
                bonus_reward -= 10.0
            else:
                bonus_reward -= 4.0

        return base_score + bonus_reward

    def step(self, action_idx: int) -> Tuple[GameState, float, bool, dict]:
        if action_idx not in self.idx_to_action:
            raise ValueError(f"Invalid action index: {action_idx}")

        action = self.idx_to_action[action_idx]
        reward = 0.0
        done = False
        info = {"action_type": action.kind.name}

        if action.kind == ActionType.ROLL:
            if self.state.rolls_left <= 0:
                raise ValueError("No rolls left!")
            self.roll_dice(np.zeros(5, dtype=bool))
            info["dice_rolled"] = True

        elif action.kind == ActionType.HOLD:
            if self.state.rolls_left <= 0:
                raise ValueError("No rolls left!")
            if not np.any(self.state.current_dice):
                raise ValueError("Cannot hold empty dice!")
            self.roll_dice(action.data)
            info["dice_held"] = np.where(action.data)[0].tolist()

        elif action.kind == ActionType.SCORE:
            category = action.data
            if self.state.score_sheet[category] is not None:
                raise ValueError(f"Category {category} already filled!")
            if not np.any(self.state.current_dice):
                raise ValueError("Cannot score empty dice!")

            points = self.calc_score(category, self.state.current_dice)
            shaped_reward = self.calc_strategic_reward(category, points)
            reward = float(shaped_reward)

            self.state.score_sheet[category] = points
            info["category_scored"] = category.name
            info["points_scored"] = points

            if all(score is not None for score in self.state.score_sheet.values()):
                done = True
                bonus = self.calc_upper_bonus()
                reward += bonus
                info["upper_bonus"] = bonus
                info["final_score"] = (
                    sum(score for score in self.state.score_sheet.values() if score)
                    + bonus
                )

            self.state.current_dice = np.zeros(5, dtype=int)
            self.state.rolls_left = 3

        return self.state, reward, done, info

    def render(self) -> str:
        lines = []
        dice_vals = self.state.current_dice
        dice_str = " ".join(str(d) if d > 0 else "-" for d in dice_vals)
        rolls = self.state.rolls_left
        lines.append(f"Dice: [{dice_str}] (rolls left: {rolls})")

        lines.append("\nScore sheet:")
        for cat in YahtzeeCategory:
            score = self.state.score_sheet[cat]
            lines.append(f"{cat.name}: {score if score is not None else '-'}")

        bonus = self.calc_upper_bonus()
        if bonus > 0:
            lines.append(f"\nUpper Bonus: +{bonus}")

        return "\n".join(lines)