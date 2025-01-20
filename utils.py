from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


def plot_training_progress(
    rewards: List[float], window: int = 100, title: Optional[str] = None
) -> None:
    """Plot training rewards with moving average."""
    plt.figure(figsize=(10, 5))

    # Raw rewards
    plt.plot(rewards, alpha=0.3, color="blue", label="Raw Rewards")

    # Moving average
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.plot(
        range(window - 1, len(rewards)),
        moving_avg,
        color="red",
        label=f"{window}-Episode Moving Average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title or "Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_game_state(state, env) -> None:
    """Visualize current game state with dice and scoresheet."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot dice
    dice_values = state.current_dice
    dice_colors = ["lightblue" if v > 0 else "lightgray" for v in dice_values]
    ax1.bar(
        range(1, 6),
        dice_values,
        color=dice_colors,
        tick_label=[f"Die {i}" for i in range(1, 6)],
    )
    ax1.set_title(f"Current Dice (Rolls Left: {state.rolls_left})")
    ax1.set_ylim(0, 6)

    # Plot scoresheet
    categories = []
    scores = []
    colors = []

    for cat in env.state.score_sheet:
        score = env.state.score_sheet[cat]
        categories.append(cat.name)
        scores.append(score if score is not None else 0)
        colors.append("lightgreen" if score is not None else "lightgray")

    y_pos = np.arange(len(categories))
    ax2.barh(y_pos, scores, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories)
    ax2.set_title("Scoresheet")

    # Add upper section total and bonus
    upper_total = env.calc_upper_score()
    bonus_text = f"Upper Total: {upper_total}"
    if upper_total >= 63:
        bonus_text += " (+35 Bonus!)"
    plt.figtext(0.98, 0.02, bonus_text, ha="right")

    plt.tight_layout()
    plt.show()


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
            elif action.kind == env.ActionType.HOLD_MASK:
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
