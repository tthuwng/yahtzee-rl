import os
from typing import List, Optional

import gradio as gr
import numpy as np
import torch

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import IDX_TO_ACTION, ActionType, YahtzeeCategory, YahtzeeEnv


def get_mtime(path: str) -> float:
    """Get modification time for a file."""
    return os.path.getmtime(path)


def find_latest_model(models_dir: str = "models") -> Optional[str]:
    """Find the latest model file in the models directory."""
    if not os.path.exists(models_dir):
        return None
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not model_files:
        return None
    # Sort by modification time
    model_files.sort(key=lambda x: get_mtime(os.path.join(models_dir, x)), reverse=True)
    return os.path.join(models_dir, model_files[0])


def list_available_models(models_dir: str = "models") -> List[str]:
    """List all available model files in the models directory."""
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith(".pth")]


def format_score_line(cat_name: str, score: Optional[int]) -> str:
    """Format a score line with proper alignment."""
    score_str = str(score) if score is not None else "-"
    return f"{cat_name:<12} {score_str:>5}"


def format_action_hold(held: List[int], held_values: List[int], q_value: float) -> str:
    """Format a hold action with proper line length."""
    pairs = zip(held, held_values)
    held_str = ", ".join(f"{pos}({val})" for pos, val in pairs)
    return f"Hold {held_str} (EV: {q_value:.1f})"


def format_straight(straight: List[int]) -> str:
    """Format a straight combination."""
    nums = map(str, straight)
    return f"Straight ({'-'.join(nums)})"


def format_score_sheet_value() -> str:
    """Return default score sheet string."""
    cats = [
        "ONES=None",
        "TWOS=None",
        "THREES=None",
        "FOURS=None",
        "FIVES=None",
        "SIXES=None",
        "THREE_OF_A_KIND=None",
        "FOUR_OF_A_KIND=None",
        "FULL_HOUSE=None",
        "SMALL_STRAIGHT=None",
        "LARGE_STRAIGHT=None",
        "YAHTZEE=None",
        "CHANCE=None",
    ]
    return ",".join(cats)


def format_distribution_line(start: float, end: float, count: int, bar: str) -> str:
    """Format a single line of the distribution display."""
    return f"{start:.0f}-{end:.0f}: {bar} ({count} games)"


def format_distribution(bins: np.ndarray, edges: np.ndarray) -> str:
    """Format score distribution as text."""
    max_bin = max(bins)
    lines = []
    for i in range(len(bins)):
        bar_len = int(bins[i] / max_bin * 20)
        bar = "█" * bar_len
        line = format_distribution_line(
            start=edges[i], end=edges[i + 1], count=bins[i], bar=bar
        )
        lines.append(line)
    return "\n".join(lines)


# Try to find a model file
DEFAULT_PATH = "models"
DEFAULT_NAME = "yahtzee_run_20250212_051636_best_eval162.pth"
DEFAULT_MODEL = os.path.join(DEFAULT_PATH, DEFAULT_NAME)
model_path = DEFAULT_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for model management
agent = None
demo_env = None
demo_encoder = None


def load_model(model_name: str) -> str:
    """Load a model and return status message."""
    global agent, demo_env, demo_encoder

    if not model_name:
        return "No model selected."

    model_path = os.path.join(DEFAULT_PATH, model_name)
    if not os.path.exists(model_path):
        return f"Model file not found: {model_path}"

    # Prepare environment and encoder first
    demo_env = YahtzeeEnv()
    demo_encoder = StateEncoder(use_opponent_value=False)  # Basic feature set

    # Create a dummy state to get encoder output size
    dummy_state = demo_env.reset()
    dummy_vec = demo_encoder.encode(dummy_state)
    state_size = len(dummy_vec)

    # Create agent with matching architecture
    agent = YahtzeeAgent(
        state_size=state_size,
        action_size=46,
        device=device,
    )

    try:
        agent.load(model_path)
        return f"Successfully loaded model: {model_name}"
    except Exception as e:
        return f"Error loading model: {e}"


# Initial model load
load_status = load_model(DEFAULT_NAME)
print(load_status)


def format_game_state(state) -> str:
    """Format game state into a nice text display."""
    output = []

    # Dice display
    dice_values = state.current_dice
    dice_str = " ".join(f"[{d}]" if d > 0 else "[ ]" for d in dice_values)
    output.append(f"\nDice (Rolls Left: {state.rolls_left})")
    output.append("Positions: [1] [2] [3] [4] [5]")
    output.append(f"Values:   {dice_str}")

    # Score board
    output.append("\nScore Board:")
    output.append("─" * 20)
    output.append(f"{'Category':<12} Score")
    output.append("─" * 20)

    # Upper section
    upper_total = 0
    for cat in [
        YahtzeeCategory.ONES,
        YahtzeeCategory.TWOS,
        YahtzeeCategory.THREES,
        YahtzeeCategory.FOURS,
        YahtzeeCategory.FIVES,
        YahtzeeCategory.SIXES,
    ]:
        score = state.score_sheet[cat]
        if score is not None:
            upper_total += score
        output.append(format_score_line(cat.name, score))

    # Upper bonus
    bonus = 35 if upper_total >= 63 else 0
    output.append("─" * 20)
    output.append(format_score_line("Sum", upper_total))
    output.append(format_score_line("Bonus", bonus))
    output.append("─" * 20)

    # Lower section
    lower_total = 0
    for cat in [
        YahtzeeCategory.THREE_OF_A_KIND,
        YahtzeeCategory.FOUR_OF_A_KIND,
        YahtzeeCategory.FULL_HOUSE,
        YahtzeeCategory.SMALL_STRAIGHT,
        YahtzeeCategory.LARGE_STRAIGHT,
        YahtzeeCategory.YAHTZEE,
        YahtzeeCategory.CHANCE,
    ]:
        score = state.score_sheet[cat]
        if score is not None:
            lower_total += score
        output.append(format_score_line(cat.name, score))

    # Total
    total = upper_total + bonus + lower_total
    output.append("─" * 20)
    output.append(format_score_line("TOTAL", total))

    # Show combinations if any dice showing
    if any(dice_values):
        counts = np.bincount(dice_values)[1:] if any(dice_values) else []
        combinations = []
        if any(c >= 3 for c in counts):
            val = next(i + 1 for i, c in enumerate(counts) if c >= 3)
            combinations.append(f"Three of a Kind ({val}s)")
        if any(c >= 4 for c in counts):
            val = next(i + 1 for i, c in enumerate(counts) if c >= 4)
            combinations.append(f"Four of a Kind ({val}s)")
        if any(c == 5 for c in counts):
            val = next(i + 1 for i, c in enumerate(counts) if c == 5)
            combinations.append(f"Yahtzee! ({val}s)")
        if any(c == 3 for c in counts) and any(c == 2 for c in counts):
            three_val = next(i + 1 for i, c in enumerate(counts) if c == 3)
            two_val = next(i + 1 for i, c in enumerate(counts) if c == 2)
            combinations.append(f"Full House ({three_val}s over {two_val}s)")
        sorted_unique = np.unique(dice_values)
        for straight in [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]:
            if all(x in sorted_unique for x in straight):
                combinations.append(format_straight(straight))
                break
        if len(sorted_unique) == 5:
            has_small = all(x in sorted_unique for x in [1, 2, 3, 4, 5])
            has_large = all(x in sorted_unique for x in [2, 3, 4, 5, 6])
            if has_small or has_large:
                straight = [1, 2, 3, 4, 5] if has_small else [2, 3, 4, 5, 6]
                combinations.append(format_straight(straight))

        if combinations:
            output.append("\nPossible Combinations:")
            for combo in combinations:
                output.append(f"• {combo}")

    return "\n".join(output)


def format_action(action_idx: int, q_value: float, state) -> str:
    """Format an action and its Q-value nicely."""
    action = IDX_TO_ACTION[action_idx]
    if action.kind == ActionType.ROLL:
        return f"Roll all dice (EV: {q_value:.1f})"
    elif action.kind == ActionType.HOLD:
        held = [i + 1 for i, hold in enumerate(action.data) if hold]
        if held:
            held_values = [state.current_dice[i - 1] for i in held]
            return format_action_hold(held, held_values, q_value)
        return f"Roll all dice (EV: {q_value:.1f})"
    else:
        points = demo_env.calc_score(action.data, state.current_dice)
        return f"Score {action.data.name} for {points} points (EV: {q_value:.1f})"


def simulate_game_interface() -> str:
    """Run a full game with the agent, returns text logs."""
    text_log = []
    state = demo_env.reset()
    done = False
    agent.eval()
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    total_reward = 0
    turn = 1

    while not done:
        text_log.append(f"\n=== Turn {turn} | Score: {total_reward:.0f} ===")
        text_log.append(format_game_state(state))

        vec = demo_encoder.encode(state)
        valid_actions = demo_env.get_valid_actions()
        if not valid_actions:
            break

        # Get Q-values and select best action
        q_values = agent.get_q_values(vec)
        mask = np.full(agent.action_size, float("-inf"))
        mask[valid_actions] = 0
        q_values = q_values + mask

        # Get top 3 actions
        valid_q = [(i, q_values[i]) for i in valid_actions]
        valid_q.sort(key=lambda x: x[1], reverse=True)
        top_actions = valid_q[:3]

        text_log.append("\nTop Actions:")
        for i, (action_idx, value) in enumerate(top_actions, 1):
            text_log.append(f"{i}. {format_action(action_idx, value, state)}")

        # Take best action
        action_idx = top_actions[0][0]
        action = IDX_TO_ACTION[action_idx]

        text_log.append("\nAgent's Decision:")
        text_log.append(format_action(action_idx, top_actions[0][1], state))

        state, reward, done, _ = demo_env.step(action_idx)
        total_reward += reward

        if action.kind == ActionType.SCORE:
            turn += 1
            text_log.append(f"\nScored {reward:.0f} points")

    text_log.append("\n=== Game Over ===")
    text_log.append(format_game_state(state))
    text_log.append(f"\nFinal Score: {total_reward:.0f}")

    agent.epsilon = old_eps
    return "\n".join(text_log)


def evaluate_performance(num_games: int = 100) -> str:
    """Run multiple games and show performance statistics."""
    output = []
    output.append(f"\nEvaluating agent performance over {num_games} games...")
    scores = []
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=False)  # Match demo encoder settings

    old_eps = agent.epsilon
    agent.epsilon = 0.02

    for i in range(num_games):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action_idx = agent.select_action(state_vec, valid_actions)
            state, reward, done, _ = env.step(action_idx)
            total_reward += reward

        scores.append(total_reward)
        if (i + 1) % 10 == 0:
            output.append(f"Completed {i + 1} games...")

    agent.epsilon = old_eps
    scores = np.array(scores)

    output.append("\nPerformance Statistics:")
    output.append(f"Mean Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    output.append(f"Median Score: {np.median(scores):.1f}")
    output.append(f"Min Score: {np.min(scores):.1f}")
    output.append(f"Max Score: {np.max(scores):.1f}")

    # Score distribution in text form
    bins = np.histogram(scores, bins=10)[0]
    bin_edges = np.histogram(scores, bins=10)[1]
    output.append("\nScore Distribution:")
    output.append(format_distribution(bins, bin_edges))

    return "\n".join(output)


def analyze_state(dice_str: str, rolls_left: int, score_dict_str: str) -> str:
    """Analyze a specific game state."""
    try:
        # Parse dice
        dice_vals = [int(x.strip()) for x in dice_str.split(",")]
        if len(dice_vals) != 5:
            return "Please provide exactly 5 dice values."
        if not all(0 <= x <= 6 for x in dice_vals):
            return "Dice values must be between 0 and 6."
    except ValueError:
        msg = "Failed parsing dice. Make sure it's comma-separated integers."
        return msg

    # Parse score dict
    score_sheet = {}
    try:
        mapping = {cat.name: cat for cat in YahtzeeCategory}
        entries = [x.strip() for x in score_dict_str.split(",")]
        for e in entries:
            if "=" in e:
                cat_str, val = e.split("=")
                cat_str = cat_str.strip()
                val = val.strip()
                cat = mapping.get(cat_str)
                if cat is None:
                    cats = ", ".join(c.name for c in YahtzeeCategory)
                    return f"Invalid category: {cat_str}. Must be one of: {cats}"
                if val.lower() == "none":
                    score_sheet[cat] = None
                else:
                    try:
                        score_sheet[cat] = int(val)
                    except ValueError:
                        return f"Invalid score value for {cat_str}: {val}"
    except Exception as e:
        return f"Error parsing score sheet: {e}"

    # Create game state
    state = demo_env.state.__class__(
        current_dice=np.array(dice_vals, dtype=int),
        rolls_left=int(rolls_left),
        score_sheet=score_sheet,
    )

    # Get analysis
    output = []
    output.append("Current Game State:")
    output.append(format_game_state(state))

    # Get Q-values
    vec = demo_encoder.encode(state)
    demo_env.state = state  # Temporarily set state to get valid actions
    valid_actions = demo_env.get_valid_actions()
    q_values = agent.get_q_values(vec)

    # Mask invalid actions
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask

    # Sort and display actions
    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)

    output.append("\nRanked Actions:")
    for i, (action_idx, value) in enumerate(valid_q, 1):
        output.append(f"{i}. {format_action(action_idx, value, state)}")

    return "\n".join(output)


def create_ui() -> gr.Blocks:
    """Create the Gradio UI interface."""
    demo = gr.Blocks(title="Yahtzee AI Assistant")

    with demo:
        gr.Markdown("""
        # Yahtzee AI Assistant
        
        Welcome! This AI has been trained using Deep RL to play Yahtzee.
        You can:
        1. Watch it play a full game
        2. Analyze specific game states
        3. Evaluate its performance
        
        Choose a mode below to get started.
        """)

        # Add model selection at the top
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=list_available_models(),
                    value=DEFAULT_NAME,
                    label="Select Model",
                    info="Choose which trained model to use",
                )
                model_status = gr.Textbox(
                    label="Model Status",
                    value=load_status,
                    interactive=False,
                )
                reload_btn = gr.Button("Reload Model", variant="secondary")

            reload_btn.click(
                fn=load_model,
                inputs=[model_dropdown],
                outputs=model_status,
            )

        with gr.Tab("Watch Game"):
            with gr.Row():
                sim_btn = gr.Button("Run Game Simulation", variant="primary")
                sim_output = gr.Textbox(
                    label="Game Progress",
                    value="Click 'Run Game Simulation' to watch.",
                    lines=25,
                    max_lines=25,
                )
            sim_btn.click(fn=simulate_game_interface, outputs=sim_output)

        with gr.Tab("Analyze Position"):
            with gr.Row():
                with gr.Column():
                    dice_help = "Enter 5 comma-separated values (0-6), e.g. '1,2,3,4,5'"
                    dice_in = gr.Textbox(
                        label="Dice Values",
                        placeholder=dice_help,
                        value="0,0,0,0,0",
                    )
                    rolls_in = gr.Number(
                        label="Rolls Left",
                        value=3,
                        minimum=0,
                        maximum=3,
                        step=1,
                    )
                    score_in = gr.Textbox(
                        label="Score Sheet",
                        placeholder="Format: ONES=None, TWOS=8, etc.",
                        value=format_score_sheet_value(),
                    )
                    analyze_btn = gr.Button(
                        "Analyze Position",
                        variant="primary",
                    )

                analyze_out = gr.Textbox(
                    label="Analysis Results",
                    value="Enter game state and click 'Analyze'",
                    lines=25,
                    max_lines=25,
                )

            analyze_btn.click(
                fn=analyze_state,
                inputs=[dice_in, rolls_in, score_in],
                outputs=analyze_out,
            )

        with gr.Tab("Performance Stats"):
            with gr.Row():
                with gr.Column():
                    num_games = gr.Number(
                        label="Number of Games",
                        value=100,
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
                    eval_btn = gr.Button("Run Evaluation", variant="primary")

                eval_out = gr.Textbox(
                    label="Evaluation Results",
                    value="Choose games and click 'Run Evaluation'",
                    lines=25,
                    max_lines=25,
                )

            eval_btn.click(
                fn=evaluate_performance, inputs=[num_games], outputs=eval_out
            )

    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch()
