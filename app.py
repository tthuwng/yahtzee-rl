from typing import Optional, Tuple, List
import os
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import glob
import time

from play import load_agent, simulate_game, show_action_values, evaluate_performance
from env import YahtzeeEnv, YahtzeeCategory, GameState, IDX_TO_ACTION, ActionType
from encoder import StateEncoder

# Global variables to maintain state
agent = None
current_state: Optional[GameState] = None
env = YahtzeeEnv()
encoder = StateEncoder(use_opponent_value=True)

def get_available_models() -> List[str]:
    """Get list of available model files in models directory."""
    models = glob.glob("models/*.pth")
    # Sort by modification time, newest first
    models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return models

def format_model_name(path: str) -> str:
    """Format model path for display in dropdown."""
    filename = os.path.basename(path)
    # Get file stats
    size_mb = os.path.getsize(path) / (1024 * 1024)
    mtime = os.path.getmtime(path)
    mtime_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
    
    # Extract evaluation score if present
    score = ""
    if "eval" in filename:
        score = f" (Eval: {filename.split('eval')[-1].split('.')[0]})"
    elif "score" in filename:
        score = f" (Score: {filename.split('score')[-1].split('.')[0]})"
        
    return f"{filename}{score} - {size_mb:.1f}MB - {mtime_str}"

def format_dice_state(dice_values: np.ndarray, rolls_left: int) -> str:
    """Format dice state for display."""
    dice_str = " ".join(f"[{d}]" if d > 0 else "[ ]" for d in dice_values)
    return f"""
Dice (Rolls Left: {rolls_left})
Positions: [1] [2] [3] [4] [5]
Values:   {dice_str}
"""

def format_score_sheet(state: GameState) -> str:
    """Format score sheet for display."""
    lines = ["Score Board:", "─" * 20, f"{'Category':<12} Score", "─" * 20]
    
    # Upper section
    for cat in [YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
               YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES]:
        score = state.score_sheet[cat]
        lines.append(f"{cat.name:<12} {score if score is not None else '-':>5}")
    
    # Calculate upper section total and bonus
    upper_total = sum(state.score_sheet[cat] or 0 for cat in list(state.score_sheet.keys())[:6])
    bonus = 35 if upper_total >= 63 else 0
    lines.extend([
        "─" * 20,
        f"{'Sum':<12} {upper_total:>5}",
        f"{'Bonus':<12} {bonus:>5}",
        "─" * 20
    ])
    
    # Lower section
    for cat in [YahtzeeCategory.THREE_OF_A_KIND, YahtzeeCategory.FOUR_OF_A_KIND,
               YahtzeeCategory.FULL_HOUSE, YahtzeeCategory.SMALL_STRAIGHT,
               YahtzeeCategory.LARGE_STRAIGHT, YahtzeeCategory.YAHTZEE,
               YahtzeeCategory.CHANCE]:
        score = state.score_sheet[cat]
        lines.append(f"{cat.name:<12} {score if score is not None else '-':>5}")
    
    # Total
    total = upper_total + bonus + sum(state.score_sheet[cat] or 0 for cat in [
        YahtzeeCategory.THREE_OF_A_KIND, YahtzeeCategory.FOUR_OF_A_KIND,
        YahtzeeCategory.FULL_HOUSE, YahtzeeCategory.SMALL_STRAIGHT,
        YahtzeeCategory.LARGE_STRAIGHT, YahtzeeCategory.YAHTZEE,
        YahtzeeCategory.CHANCE
    ])
    lines.extend([
        "─" * 20,
        f"{'TOTAL':<12} {total:>5.0f}"
    ])
    
    return "\n".join(lines)

def format_combinations(dice_values: np.ndarray) -> str:
    """Format possible combinations for display."""
    if not any(dice_values):
        return ""
        
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
        return "\nPossible Combinations:\n" + "\n".join(f"• {combo}" for combo in combinations)
    return ""

def load_model(model_choice: str) -> str:
    """Load a model and return status message."""
    global agent
    try:
        agent = load_agent(model_choice)
        return f"Successfully loaded model: {os.path.basename(model_choice)}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def run_simulation() -> Tuple[str, str]:
    """Run a full game simulation and return the game log."""
    if agent is None:
        return "Please load a model first.", ""
    
    env = YahtzeeEnv()
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1
    game_log = []
    
    # Store original epsilon and set to minimum for deterministic play
    old_eps = agent.epsilon
    agent.epsilon = 0.02
    
    try:
        while not done:
            # Format current state
            game_log.append(f"\n=== Turn {turn} | Score: {total_reward:.0f} ===")
            game_log.append(format_dice_state(state.current_dice, state.rolls_left))
            game_log.append(format_score_sheet(state))
            game_log.append(format_combinations(state.current_dice))
            
            # Get agent's action
            state_vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # Get Q-values and select best action
            q_values = agent.get_q_values(state_vec)
            q_values = q_values - total_reward  # Make values relative to current score
            
            # Apply action masking
            mask = np.full(agent.action_size, float("-inf"))
            mask[valid_actions] = 0
            q_values = q_values + mask
            
            # Get top actions
            valid_q = [(i, q_values[i]) for i in valid_actions]
            valid_q.sort(key=lambda x: x[1], reverse=True)
            action_idx = valid_q[0][0]
            action = IDX_TO_ACTION[action_idx]
            
            # Show decision
            game_log.append("\nAgent's Decision:")
            if action.kind == ActionType.ROLL:
                game_log.append("Rolling all dice")
            elif action.kind == ActionType.HOLD:
                held = [i + 1 for i, hold in enumerate(action.data) if hold]
                if held:
                    held_values = [state.current_dice[i-1] for i in held]
                    game_log.append(f"Holding: {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))}")
                else:
                    game_log.append("Rolling all dice")
            else:
                points = env.calc_score(action.data, state.current_dice)
                game_log.append(f"Scoring {action.data.name} for {points} points")
            
            # Take action
            state, reward, done, _ = env.step(action_idx)
            total_reward += reward
            
            if action.kind == ActionType.SCORE:
                turn += 1
        
        # Show final results
        game_log.append("\n=== Game Over ===")
        game_log.append(format_dice_state(state.current_dice, state.rolls_left))
        game_log.append(format_score_sheet(state))
        
        # Calculate final scores
        upper_scores = [state.score_sheet[cat] or 0 for cat in [
            YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES
        ]]
        upper_total = sum(upper_scores)
        bonus = 35 if upper_total >= 63 else 0
        lower_score = total_reward - upper_total - bonus
        
        game_log.append(f"\nFinal Score: {total_reward:.0f}")
        game_log.append(f"• Upper Section: {upper_total}")
        game_log.append(f"• Upper Bonus: {bonus}")
        game_log.append(f"• Lower Section: {lower_score:.0f}")
        
        agent.epsilon = old_eps
        return "\n".join(game_log), f"Final Score: {total_reward:.0f}"
        
    except Exception as e:
        return f"Error during simulation: {str(e)}", ""

def analyze_state(state_input: Optional[str] = None) -> Tuple[str, str]:
    """Analyze current state or create new state."""
    global current_state
    
    if agent is None:
        return "Please load a model first.", ""
        
    if state_input == "new":
        current_state = env.reset()
    elif current_state is None:
        current_state = env.reset()
    
    # Format current state
    state_display = []
    state_display.append(format_dice_state(current_state.current_dice, current_state.rolls_left))
    state_display.append(format_score_sheet(current_state))
    state_display.append(format_combinations(current_state.current_dice))
    
    # Get action values
    state_vec = encoder.encode(current_state)
    valid_actions = env.get_valid_actions()
    
    if not valid_actions:
        return "No valid actions available.", "\n".join(state_display)
        
    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask
    
    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)
    
    # Format action values
    action_display = ["Top Actions and Their Expected Values:"]
    for i, (action_idx, value) in enumerate(valid_q[:5], 1):
        action = IDX_TO_ACTION[action_idx]
        if action.kind == ActionType.ROLL:
            action_display.append(f"{i}. Roll all dice (EV: {value:.1f})")
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                held_values = [current_state.current_dice[i-1] for i in held]
                action_display.append(f"{i}. Hold {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))} (EV: {value:.1f})")
            else:
                action_display.append(f"{i}. Roll all dice (EV: {value:.1f})")
        else:
            points = env.calc_score(action.data, current_state.current_dice)
            action_display.append(f"{i}. Score {action.data.name} for {points} points (EV: {value:.1f})")
    
    return "\n".join(action_display), "\n".join(state_display)

def take_action(action_num: int) -> Tuple[str, str]:
    """Take the specified action in calculation mode."""
    global current_state
    
    if agent is None:
        return "Please load a model first.", ""
    if current_state is None:
        return "Please start a new state first.", ""
        
    # Get valid actions and their values
    state_vec = encoder.encode(current_state)
    valid_actions = env.get_valid_actions()
    
    if not valid_actions:
        return "No valid actions available.", ""
        
    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask
    
    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)
    
    if not 1 <= action_num <= len(valid_q):
        return f"Invalid action number. Please choose between 1 and {len(valid_q)}.", ""
        
    # Take the action
    action_idx = valid_q[action_num - 1][0]
    current_state, reward, done, _ = env.step(action_idx)
    
    if done:
        result = "Game Over!"
        current_state = None
    else:
        result = f"Action taken, received reward: {reward:.1f}"
        
    # Get updated state analysis
    action_analysis, state_display = analyze_state()
    return f"{result}\n\n{action_analysis}", state_display

def run_performance_analysis() -> Tuple[str, None]:
    """Run performance analysis and return statistics."""
    if agent is None:
        return "Please load a model first.", None
        
    try:
        # Run evaluation
        scores = []
        old_eps = agent.epsilon
        agent.epsilon = 0.02
        
        for _ in range(100):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state_vec = encoder.encode(state)
                valid_actions = env.get_valid_actions()
                
                # Skip if no valid actions
                if not valid_actions:
                    break
                    
                # Get Q-values and select best action
                q_values = agent.get_q_values(state_vec)
                
                # Apply action masking
                mask = np.full(agent.action_size, float("-inf"))
                mask[valid_actions] = 0
                q_values = q_values + mask
                
                # Get top action
                valid_q = [(i, q_values[i]) for i in valid_actions]
                valid_q.sort(key=lambda x: x[1], reverse=True)
                action_idx = valid_q[0][0]
                
                # Take action
                state, reward, done, _ = env.step(action_idx)
                total_reward += reward
                
            scores.append(total_reward)
            
        agent.epsilon = old_eps
        
        # Calculate statistics
        scores = np.array(scores)
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        percentile_25 = np.percentile(scores, 25)
        percentile_75 = np.percentile(scores, 75)
        
        # Format statistics with ASCII visualization
        stats = f"""Performance Statistics (100 games)

Key Metrics:
• Mean Score:   {mean_score:.1f} ± {std_score:.1f}
• Median Score: {median_score:.1f}
• Min Score:    {min_score:.1f}
• Max Score:    {max_score:.1f}

Score Distribution:
• 25th percentile: {percentile_25:.1f}
• 75th percentile: {percentile_75:.1f}
• IQR:            {(percentile_75 - percentile_25):.1f}

Score Brackets:"""
        
        # Add score distribution in brackets
        brackets = [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, float('inf'))]
        for low, high in brackets:
            count = np.sum((scores >= low) & (scores < high))
            percentage = (count / len(scores)) * 100
            high_str = f"{high:.0f}" if high != float('inf') else "inf"
            bar = "█" * int(percentage / 2)  # Each █ represents 2%
            stats += f"\n• {low:3.0f}-{high_str:>3}: {bar} {count:3.0f} games ({percentage:4.1f}%)"
            
        return stats, None
        
    except Exception as e:
        return f"Error during performance analysis: {str(e)}", None

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(title="Yahtzee RL Challenge") as interface:
        gr.Markdown("""# Yahtzee RL Challenge
        
This interface demonstrates a Deep Q-Learning agent trained to play Yahtzee. The agent makes decisions by evaluating the expected value of each possible action, considering both immediate rewards and potential future gains.

Start by selecting a trained model, then explore the different modes:
1. **Simulation Mode**: Watch the agent play a complete game
2. **Calculation Mode**: Analyze expected values for each possible action
3. **Performance Stats**: View the agent's statistics over 100 games
        """)
        
        # Model loading section
        with gr.Row():
            model_choices = get_available_models()
            model_dropdown = gr.Dropdown(
                choices=[format_model_name(m) for m in model_choices],
                value=format_model_name(model_choices[0]) if model_choices else None,
                label="Select Model",
                info="Models are sorted by newest first"
            )
            load_button = gr.Button("Load Selected Model")
            model_status = gr.Textbox(label="Status", interactive=False)
            
        # Connect the dropdown value to the actual model path
        def on_load_model(choice: str) -> str:
            if not choice:
                return "Please select a model"
            # Find the corresponding model path
            idx = [format_model_name(m) for m in model_choices].index(choice)
            return load_model(model_choices[idx])
        
        # Create tabs for different modes
        with gr.Tabs():
            # Simulation Mode
            with gr.Tab("Simulation Mode"):
                gr.Markdown("""Watch the agent play a complete game of Yahtzee. The agent will show its decision-making process for each move.""")
                sim_button = gr.Button("Start New Game")
                with gr.Row():
                    sim_output = gr.Textbox(label="Game Log", interactive=False)
                    sim_score = gr.Textbox(label="Result", interactive=False)
            
            # Calculation Mode
            with gr.Tab("Calculation Mode"):
                gr.Markdown("""Analyze the expected value of each possible action in any game state.""")
                with gr.Row():
                    new_state_button = gr.Button("New Game State")
                    action_input = gr.Number(label="Action Number", value=1, minimum=1, maximum=5, step=1)
                    take_action_button = gr.Button("Take Action")
                with gr.Row():
                    action_output = gr.Textbox(label="Action Analysis", interactive=False)
                    state_output = gr.Textbox(label="Current State", interactive=False)
            
            # Performance Stats
            with gr.Tab("Performance Stats"):
                gr.Markdown("""View detailed statistics of the agent's performance over 100 games.""")
                stats_button = gr.Button("Run Performance Analysis")
                with gr.Row():
                    stats_output = gr.Textbox(label="Statistics", interactive=False)
        
        # Connect components
        load_button.click(on_load_model, inputs=[model_dropdown], outputs=[model_status])
        sim_button.click(run_simulation, outputs=[sim_output, sim_score])
        new_state_button.click(analyze_state, inputs=[gr.Textbox(value="new", visible=False)], outputs=[action_output, state_output])
        take_action_button.click(take_action, inputs=[action_input], outputs=[action_output, state_output])
        stats_button.click(run_performance_analysis, outputs=[stats_output])
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True) 