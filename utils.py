from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


def plot_training_progress(
    rewards: List[float], window: int = 100, title: Optional[str] = None
) -> None:
    """Plot training rewards with moving average."""
    plt.close('all')  # Close all previous figures
    plt.figure(figsize=(10, 5))
    
    # Moving average
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.plot(
        range(window - 1, len(rewards)),
        moving_avg,
        color="red",
        label=f"{window}-Episode Average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(title or "Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()


def visualize_game_state(state, env) -> None:
    """Visualize current game state with dice and scoresheet."""
    # Show dice state
    dice_values = state.current_dice
    dice_str = " ".join(f"[{d}]" if d > 0 else "[ ]" for d in dice_values)
    print(f"\nDice (Rolls Left: {state.rolls_left})")
    print(f"Positions: [1] [2] [3] [4] [5]")
    print(f"Values:   {dice_str}")
    
    # Calculate scores and bonuses
    upper_scores = [env.state.score_sheet[cat] or 0 for cat in list(env.state.score_sheet.keys())[:6]]
    upper_total = sum(upper_scores)
    bonus = 35 if upper_total >= 63 else 0
    points_needed = max(0, 63 - upper_total)
    upper_remaining = sum(1 for s in upper_scores if s == 0)
    
    # Show score summary
    print("\nScore Summary:")
    print(f"Upper Section: {upper_total} ({points_needed} needed for bonus)")
    if upper_remaining > 0:
        print(f"Avg needed per category: {points_needed/upper_remaining:.1f}")
    
    # Show available categories
    print("\nAvailable Categories:")
    for cat, score in env.state.score_sheet.items():
        if score is None:
            print(f"• {cat.name}")
    
    # Show current combinations
    if any(dice_values):
        counts = np.bincount(dice_values)[1:] if any(dice_values) else []
        combinations = []
        if any(c >= 3 for c in counts):
            val = next(i+1 for i, c in enumerate(counts) if c >= 3)
            combinations.append(f"Three of a Kind ({val}s)")
        if any(c >= 4 for c in counts):
            val = next(i+1 for i, c in enumerate(counts) if c >= 4)
            combinations.append(f"Four of a Kind ({val}s)")
        if any(c == 5 for c in counts):
            val = next(i+1 for i, c in enumerate(counts) if c == 5)
            combinations.append(f"Yahtzee! ({val}s)")
        if any(c == 3 for c in counts) and any(c == 2 for c in counts):
            three_val = next(i+1 for i, c in enumerate(counts) if c == 3)
            two_val = next(i+1 for i, c in enumerate(counts) if c == 2)
            combinations.append(f"Full House ({three_val}s over {two_val}s)")
        sorted_unique = np.unique(dice_values)
        for straight in [[1,2,3,4], [2,3,4,5], [3,4,5,6]]:
            if all(x in sorted_unique for x in straight):
                combinations.append(f"Small Straight ({'-'.join(map(str, straight))})")
                break
        if len(sorted_unique) == 5 and (
            all(x in sorted_unique for x in [1,2,3,4,5]) or
            all(x in sorted_unique for x in [2,3,4,5,6])
        ):
            straight = [1,2,3,4,5] if sorted_unique[0] == 1 else [2,3,4,5,6]
            combinations.append(f"Large Straight ({'-'.join(map(str, straight))})")
        
        if combinations:
            print("\nPossible Combinations:")
            for combo in combinations:
                print(f"• {combo}")


def simulate_game(agent, env, encoder, render: bool = True) -> float:
    """Simulate a single game with visualization."""
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if render:
            clear_output(wait=True)
            visualize_game_state(state, env)

        state_vec = encoder.encode(state)
        valid_actions = env.get_valid_actions()
        action_idx = agent.select_action(state_vec, valid_actions)

        # Get action description for display
        action = env.IDX_TO_ACTION[action_idx]
        if render:
            if action.kind == env.ActionType.ROLL:
                print("\nAction: ROLL")
            elif action.kind == env.ActionType.HOLD:
                print(f"\nAction: Hold dice at positions: {action.data}")
            else:
                print(f"\nAction: Score {action.data.name}")

        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        if render:
            print(f"Reward: {reward:.1f}")
            print(f"Total Score: {total_reward:.1f}")

    if render:
        clear_output(wait=True)
        visualize_game_state(state, env)
        print(f"\nGame Over! Final Score: {total_reward:.1f}")

    return total_reward
