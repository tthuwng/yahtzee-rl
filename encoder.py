from typing import Dict, List, Optional, Tuple

import numpy as np

from yahtzee_types import Action, ActionType, GameState, YahtzeeCategory

# Action mapping constants
NUM_ACTIONS = 46  # 1 ROLL + 32 HOLD + 13 SCORE
ALL_ACTIONS = []  # Will be populated in ActionMapper
ACTION_TO_IDX = {}  # Will be populated in ActionMapper
IDX_TO_ACTION = {}  # Will be populated in ActionMapper


class StateEncoder:
    """Enhanced state encoder with yahtzee-specific strategic features."""

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

        # Calculate basic state features
        self.basic_features = (
            1  # Number of rerolls left
            + 6  # Count of each die value (1-6)
            + self.num_categories  # Player scorecard (filled categories)
            + 2  # Player upper and lower scores
            + (1 if use_opponent_value else 0)  # Opponent value (optional)
        )

        # Add strategic features
        self.strategic_features = (
            1  # Upper section bonus status
            + 1  # Upper section bonus potential
            + 5  # Key scoring probabilities
            + 3  # Pattern indicators (pairs, three of a kind, etc.)
            + 4  # Distance metrics for straights
        )

        # Total state size with strategic features
        self.state_size = self.basic_features + self.strategic_features

    def _get_dice_counts(self, dice: np.ndarray) -> np.ndarray:
        """Get counts of each dice value (1-6)."""
        return np.bincount(dice, minlength=7)[1:]  # Skip index 0

    def _get_score_summary(
        self, scores: Dict[YahtzeeCategory, Optional[int]]
    ) -> Tuple[float, float]:
        """Calculate normalized upper and lower section scores."""
        upper_score = sum(scores[cat] or 0 for cat in self.upper_cats)
        lower_score = sum(
            scores[cat] or 0 for cat in self.categories if cat not in self.upper_cats
        )

        # Normalize scores
        upper_score_norm = min(upper_score / 63.0, 1.0)  # 63 is bonus threshold
        lower_score_norm = min(
            lower_score / 200.0, 1.0
        )  # 200 is approximate max lower score

        return upper_score_norm, lower_score_norm, upper_score

    def _has_bonus(self, scores: Dict[YahtzeeCategory, Optional[int]]) -> float:
        """Check if upper section bonus is achieved or close."""
        upper_score = sum(scores[cat] or 0 for cat in self.upper_cats)
        if upper_score >= 63:
            return 1.0  # Bonus achieved
        elif upper_score >= 50:
            return 0.75  # Very close to bonus
        elif upper_score >= 40:
            return 0.5  # Getting close to bonus
        elif upper_score >= 30:
            return 0.25  # Still work to do
        return 0.0

    def _bonus_potential(self, scores: Dict[YahtzeeCategory, Optional[int]]) -> float:
        """Calculate potential to still achieve the upper section bonus."""
        upper_score = sum(scores[cat] or 0 for cat in self.upper_cats)
        filled_cats = sum(1 for cat in self.upper_cats if scores[cat] is not None)

        # If bonus already achieved or all categories filled
        if upper_score >= 63 or filled_cats == 6:
            return 0.0

        # Calculate needed average per remaining category
        remaining_cats = 6 - filled_cats
        needed_points = 63 - upper_score
        if remaining_cats == 0:
            return 0.0

        avg_needed = needed_points / remaining_cats

        # Scale from 0-1 based on feasibility
        # 3 per category is the expected average, 5 is very good
        if avg_needed <= 3:
            return 1.0  # Very achievable
        elif avg_needed <= 4:
            return 0.75  # Fairly achievable
        elif avg_needed <= 5:
            return 0.5  # Challenging
        elif avg_needed <= 6:
            return 0.25  # Very difficult
        return 0.0  # Virtually impossible

    def _pattern_indicators(self, dice: np.ndarray) -> List[float]:
        """Create indicators for common dice patterns."""
        if not np.any(dice):
            return [0.0, 0.0, 0.0]  # No dice showing

        counts = self._get_dice_counts(dice)
        max_count = max(counts) if len(counts) > 0 else 0
        unique_vals = np.count_nonzero(counts)

        # Pairs or better
        has_pair = 1.0 if max_count >= 2 else 0.0

        # Three of a kind or better
        has_three = 1.0 if max_count >= 3 else 0.0

        # Is a full house or close
        is_full_house = 0.0
        if max_count >= 3:
            # Check if there's also a pair
            counts_copy = counts.copy()
            max_index = np.argmax(counts_copy)
            counts_copy[max_index] = 0  # Remove the three of a kind
            if max(counts_copy) >= 2:
                is_full_house = 1.0  # Full house
            elif max(counts_copy) == 1:
                is_full_house = 0.5  # One away from full house

        return [has_pair, has_three, is_full_house]

    def _straight_metrics(self, dice: np.ndarray) -> List[float]:
        """Calculate how close the dice are to achieving straights."""
        if not np.any(dice):
            return [0.0, 0.0, 0.0, 0.0]  # No dice showing

        unique_vals = set(dice) - {0}  # Remove any zeros

        # Small straight patterns
        small_patterns = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
        small_straight_scores = []

        for pattern in small_patterns:
            matches = sum(1 for val in pattern if val in unique_vals)
            small_straight_scores.append(matches / 4.0)

        # Large straight patterns
        large_patterns = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        large_straight_scores = []

        for pattern in large_patterns:
            matches = sum(1 for val in pattern if val in unique_vals)
            large_straight_scores.append(matches / 5.0)

        # Return best small and large straight scores
        return [
            max(small_straight_scores) if small_straight_scores else 0.0,
            small_straight_scores[0] if small_straight_scores else 0.0,
            small_straight_scores[1] if len(small_straight_scores) > 1 else 0.0,
            small_straight_scores[2] if len(small_straight_scores) > 2 else 0.0,
        ]

    def _key_probabilities(self, state: GameState) -> List[float]:
        """Calculate probabilities for achieving key combinations."""
        dice = state.current_dice
        rolls_left = state.rolls_left

        if not np.any(dice) or rolls_left == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        counts = self._get_dice_counts(dice)
        max_count = max(counts) if len(counts) > 0 else 0

        # Probability of getting Yahtzee
        yahtzee_prob = 0.0
        if max_count == 5:
            yahtzee_prob = 1.0  # Already have it
        elif max_count == 4:
            yahtzee_prob = min(1.0, 0.33 * rolls_left)  # 1/6 per roll of needed number
        elif max_count == 3:
            yahtzee_prob = min(1.0, 0.08 * rolls_left)  # Lower probability
        elif max_count == 2:
            yahtzee_prob = min(1.0, 0.02 * rolls_left)  # Very low probability

        # Probability of getting a large straight
        unique_vals = set(dice) - {0}  # Remove any zeros
        large_straight_prob = 0.0

        if len(unique_vals) == 5 and (
            all(x in unique_vals for x in [1, 2, 3, 4, 5])
            or all(x in unique_vals for x in [2, 3, 4, 5, 6])
        ):
            large_straight_prob = 1.0  # Already have it
        elif len(unique_vals) == 4:
            # Need just one specific number
            large_straight_prob = min(1.0, 0.3 * rolls_left)
        elif len(unique_vals) == 3:
            # Need two specific numbers
            large_straight_prob = min(1.0, 0.1 * rolls_left)

        # Probability of getting a small straight
        small_straight_prob = 0.0
        for pattern in [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]:
            matches = sum(1 for val in pattern if val in unique_vals)
            if matches == 4:
                small_straight_prob = 1.0  # Already have it
                break
            elif matches == 3:
                small_straight_prob = max(
                    small_straight_prob, min(1.0, 0.4 * rolls_left)
                )
            elif matches == 2:
                small_straight_prob = max(
                    small_straight_prob, min(1.0, 0.15 * rolls_left)
                )

        # Probability of getting three of a kind
        three_kind_prob = 0.0
        if max_count >= 3:
            three_kind_prob = 1.0  # Already have it
        elif max_count == 2:
            three_kind_prob = min(1.0, 0.6 * rolls_left)  # Good chance
        else:
            three_kind_prob = min(1.0, 0.3 * rolls_left)  # Lower chance

        # Probability of getting a full house
        full_house_prob = 0.0
        counts_list = sorted(counts, reverse=True)
        if len(counts_list) >= 2 and counts_list[0] >= 3 and counts_list[1] >= 2:
            full_house_prob = 1.0  # Already have it
        elif len(counts_list) >= 2 and counts_list[0] >= 3:
            # Have three of a kind, need a pair
            full_house_prob = min(1.0, 0.5 * rolls_left)
        elif len(counts_list) >= 2 and counts_list[0] == 2 and counts_list[1] == 2:
            # Have two pairs, need to convert one to three of a kind
            full_house_prob = min(1.0, 0.4 * rolls_left)
        elif len(counts_list) >= 1 and counts_list[0] == 2:
            # Have one pair, harder to get full house
            full_house_prob = min(1.0, 0.2 * rolls_left)

        return [
            yahtzee_prob,
            large_straight_prob,
            small_straight_prob,
            three_kind_prob,
            full_house_prob,
        ]

    def encode(self, state: GameState, opponent_value: float = 0.0) -> np.ndarray:
        """Convert game state to enhanced vector representation."""
        # Initialize with zeros
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
        upper_score, lower_score, upper_raw = self._get_score_summary(state.score_sheet)
        vec[idx] = upper_score
        vec[idx + 1] = lower_score
        idx += 2

        # 5. Opponent value (if used)
        if self.use_opponent_value:
            vec[idx] = opponent_value
            idx += 1

        # 6. Upper section bonus status
        vec[idx] = self._has_bonus(state.score_sheet)
        idx += 1

        # 7. Upper section bonus potential
        vec[idx] = self._bonus_potential(state.score_sheet)
        idx += 1

        # 8. Key scoring probabilities
        prob_features = self._key_probabilities(state)
        for prob in prob_features:
            vec[idx] = prob
            idx += 1

        # 9. Pattern indicators
        pattern_features = self._pattern_indicators(state.current_dice)
        for pattern in pattern_features:
            vec[idx] = pattern
            idx += 1

        # 10. Straight metrics
        straight_features = self._straight_metrics(state.current_dice)
        for straight in straight_features:
            vec[idx] = straight
            idx += 1

        # Verify we've filled exactly the right number of features
        assert idx == self.state_size, (
            f"Feature count mismatch. Filled {idx}, expected {self.state_size}"
        )

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

    def valid_action_mask(self, valid_actions: List[int]) -> np.ndarray:
        """Create binary mask for valid actions."""
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[valid_actions] = 1.0
        return mask

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
                        hold_act = Action(ActionType.HOLD, tuple(mask))
                        valid.append(ACTION_TO_IDX[hold_act])

        # can score if dice are showing
        if np.any(state.current_dice):
            for cat in YahtzeeCategory:
                if state.score_sheet[cat] is None:
                    score_act = Action(ActionType.SCORE, cat)
                    valid.append(ACTION_TO_IDX[score_act])

        return valid
