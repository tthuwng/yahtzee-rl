import numpy as np


def visualize_game_state(state, env) -> None:
    """Visualize current game state with dice and scoresheet."""
    # Show dice state
    dice_vals = state.current_dice
    dice_str = " ".join(f"[{d}]" if d > 0 else "[ ]" for d in dice_vals)
    print(f"\nDice (Rolls Left: {state.rolls_left})")
    print("Positions: [1] [2] [3] [4] [5]")
    print(f"Values:   {dice_str}")

    # Calculate scores and bonuses
    upper_cats = list(env.state.score_sheet.keys())[:6]
    upper_scores = [env.state.score_sheet[cat] or 0 for cat in upper_cats]
    upper_total = sum(upper_scores)
    points_needed = max(0, 63 - upper_total)
    upper_remaining = sum(1 for s in upper_scores if s == 0)

    # Show score summary
    print("\nScore Summary:")
    print(f"Upper Section: {upper_total} ({points_needed} needed for bonus)")
    if upper_remaining > 0:
        avg_needed = points_needed / upper_remaining
        print(f"Avg needed per category: {avg_needed:.1f}")

    # Show available categories
    print("\nAvailable Categories:")
    for cat, score in env.state.score_sheet.items():
        if score is None:
            print(f"• {cat.name}")

    # Show current combinations
    if any(dice_vals):
        counts = np.bincount(dice_vals)[1:] if any(dice_vals) else []
        combinations = []

        # Check for three/four of a kind
        for count, name in [(3, "Three"), (4, "Four")]:
            if any(c >= count for c in counts):
                val = next(i + 1 for i, c in enumerate(counts) if c >= count)
                combinations.append(f"{name} of a Kind ({val}s)")

        # Check for yahtzee
        if any(c == 5 for c in counts):
            val = next(i + 1 for i, c in enumerate(counts) if c == 5)
            combinations.append(f"Yahtzee! ({val}s)")

        # Check for full house
        if any(c == 3 for c in counts) and any(c == 2 for c in counts):
            three_val = next(i + 1 for i, c in enumerate(counts) if c == 3)
            two_val = next(i + 1 for i, c in enumerate(counts) if c == 2)
            combinations.append(f"Full House ({three_val}s over {two_val}s)")

        # Check for straights
        sorted_unique = np.unique(dice_vals)
        straights = [
            ([1, 2, 3, 4], "Small"),
            ([2, 3, 4, 5], "Small"),
            ([3, 4, 5, 6], "Small"),
            ([1, 2, 3, 4, 5], "Large"),
            ([2, 3, 4, 5, 6], "Large"),
        ]

        for nums, kind in straights:
            if all(x in sorted_unique for x in nums):
                nums_str = "-".join(map(str, nums))
                combinations.append(f"{kind} Straight ({nums_str})")
                if kind == "Small":  # Only need one small straight
                    break

        if combinations:
            print("\nPossible Combinations:")
            for combo in combinations:
                print(f"• {combo}")
