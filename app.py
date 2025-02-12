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
from env import YahtzeeEnv, YahtzeeCategory, GameState, IDX_TO_ACTION, ActionType, NUM_ACTIONS
from encoder import StateEncoder
from dqn import YahtzeeAgent

# Global variables to maintain state
agent = None
current_state: Optional[GameState] = None
env = YahtzeeEnv()
encoder = StateEncoder(use_opponent_value=True)
current_objective = "win"  # Add global objective state

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
    upper_total = 0
    for cat in [YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
               YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES]:
        score = state.score_sheet[cat]
        upper_total += score if score is not None else 0
        lines.append(f"{cat.name:<12} {score if score is not None else '-':>5}")
    
    # Calculate upper section bonus
    bonus = 35 if upper_total >= 63 else 0
    lines.extend([
        "─" * 20,
        f"{'Sum':<12} {upper_total:>5}",
        f"{'Bonus':<12} {bonus:>5}",
        "─" * 20
    ])
    
    # Lower section
    lower_total = 0
    for cat in [YahtzeeCategory.THREE_OF_A_KIND, YahtzeeCategory.FOUR_OF_A_KIND,
               YahtzeeCategory.FULL_HOUSE, YahtzeeCategory.SMALL_STRAIGHT,
               YahtzeeCategory.LARGE_STRAIGHT, YahtzeeCategory.YAHTZEE,
               YahtzeeCategory.CHANCE]:
        score = state.score_sheet[cat]
        lower_total += score if score is not None else 0
        lines.append(f"{cat.name:<12} {score if score is not None else '-':>5}")
    
    # Total
    total = upper_total + bonus + lower_total
    lines.extend([
        "─" * 20,
        f"{'TOTAL':<12} {total:>5}"
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

def load_model(model_path: str) -> Tuple[str, None]:
    """Load a model from the specified path."""
    global agent, current_objective
    
    try:
        # Create encoder instance to get state size
        encoder = StateEncoder(use_opponent_value=(current_objective == "win"))
        
        # Determine device - use CPU if CUDA not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize agent with correct parameters
        agent = YahtzeeAgent(
            state_size=encoder.state_size,
            action_size=NUM_ACTIONS,
            batch_size=512,
            gamma=0.997,
            learning_rate=5e-5,
            target_update=50,
            device=device  # Pass device explicitly
        )
        
        # Load model with proper device mapping
        agent.load(model_path)
        agent.eval()
        
        return f"Successfully loaded model from {model_path} (using {device})", None
    except Exception as e:
        return f"Error loading model: {str(e)}", None

def set_objective(obj: str) -> Tuple[str, None]:
    """Set the current evaluation objective."""
    global current_objective
    current_objective = obj
    return f"Evaluation objective set to: {obj}", None

def run_simulation() -> Tuple[str, str]:
    """Run a full game simulation and return the game log."""
    global current_objective
    
    if agent is None:
        return "Please load a model first.", ""
    
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=(current_objective == "win"))
    state = env.reset()
    total_reward = 0  # For training rewards
    done = False
    turn = 1
    game_log = []
    
    # Store original epsilon and set to minimum for deterministic play
    old_eps = agent.epsilon
    agent.epsilon = 0.02
    
    try:
        while not done:
            # Calculate actual game score
            upper_total = sum(state.score_sheet[cat] or 0 for cat in [
                YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
                YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES
            ])
            bonus = 35 if upper_total >= 63 else 0
            lower_total = sum(state.score_sheet[cat] or 0 for cat in [
                YahtzeeCategory.THREE_OF_A_KIND, YahtzeeCategory.FOUR_OF_A_KIND,
                YahtzeeCategory.FULL_HOUSE, YahtzeeCategory.SMALL_STRAIGHT,
                YahtzeeCategory.LARGE_STRAIGHT, YahtzeeCategory.YAHTZEE,
                YahtzeeCategory.CHANCE
            ])
            actual_score = upper_total + bonus + lower_total
            
            # Format current state
            game_log.append(f"\n=== Turn {turn} | Score: {actual_score} ===")
            game_log.append(format_dice_state(state.current_dice, state.rolls_left))
            game_log.append(format_score_sheet(state))
            game_log.append(format_combinations(state.current_dice))
            
            # Get agent's action
            state_vec = encoder.encode(
                state, 
                opponent_value=0.5 if current_objective == "win" else 0.0
            )
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # Get Q-values and select best action
            q_values = agent.get_q_values(state_vec)
            
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
        upper_total = sum(state.score_sheet[cat] or 0 for cat in [
            YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES
        ])
        bonus = 35 if upper_total >= 63 else 0
        lower_total = sum(state.score_sheet[cat] or 0 for cat in [
            YahtzeeCategory.THREE_OF_A_KIND, YahtzeeCategory.FOUR_OF_A_KIND,
            YahtzeeCategory.FULL_HOUSE, YahtzeeCategory.SMALL_STRAIGHT,
            YahtzeeCategory.LARGE_STRAIGHT, YahtzeeCategory.YAHTZEE,
            YahtzeeCategory.CHANCE
        ])
        actual_score = upper_total + bonus + lower_total
        
        game_log.append(f"\nFinal Score: {actual_score}")
        game_log.append(f"• Upper Section: {upper_total}")
        game_log.append(f"• Upper Bonus: {bonus}")
        game_log.append(f"• Lower Section: {lower_total}")
        
        agent.epsilon = old_eps
        return "\n".join(game_log), f"Final Score: {actual_score}"
        
    except Exception as e:
        return f"Error during simulation: {str(e)}", ""

def analyze_state(state_input: Optional[str] = None) -> Tuple[str, str]:
    """Analyze current state or create new state."""
    global current_state, env
    
    if agent is None:
        return "Please load a model first.", ""
        
    if state_input == "new":
        env = YahtzeeEnv()  # Reset environment
        current_state = env.reset()
    elif current_state is None:
        env = YahtzeeEnv()  # Reset environment
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
        return "No valid actions available.", format_score_sheet(current_state)
        
    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask
    
    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)
    
    if not 1 <= action_num <= len(valid_q):
        return f"Invalid action number. Please choose between 1 and {len(valid_q)}.", format_dice_state(current_state.current_dice, current_state.rolls_left) + "\n" + format_score_sheet(current_state)
        
    # Take the action
    action_idx = valid_q[action_num - 1][0]
    action = IDX_TO_ACTION[action_idx]
    old_state = current_state
    current_state, reward, done, _ = env.step(action_idx)
    
    # Format action result message
    if action.kind == ActionType.ROLL:
        action_msg = "Rolling all dice"
    elif action.kind == ActionType.HOLD:
        held = [i + 1 for i, hold in enumerate(action.data) if hold]
        if held:
            held_values = [old_state.current_dice[i-1] for i in held]
            action_msg = f"Holding positions {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))}"
        else:
            action_msg = "Rolling all dice"
    else:
        points = env.calc_score(action.data, old_state.current_dice)
        action_msg = f"Scoring {action.data.name} for {points} points"
    
    result = [
        f"Action taken: {action_msg}",
        f"Reward: {reward:.1f}",
        ""
    ]
    
    if done:
        result.append("Game Over!")
        current_state = None
    else:
        # Get updated state analysis
        state_vec = encoder.encode(current_state)
        valid_actions = env.get_valid_actions()
        q_values = agent.get_q_values(state_vec)
        mask = np.full(agent.action_size, float("-inf"))
        mask[valid_actions] = 0
        q_values = q_values + mask
        
        valid_q = [(i, q_values[i]) for i in valid_actions]
        valid_q.sort(key=lambda x: x[1], reverse=True)
        
        result.append("Top Actions and Their Expected Values:")
        for i, (action_idx, value) in enumerate(valid_q[:5], 1):
            action = IDX_TO_ACTION[action_idx]
            if action.kind == ActionType.ROLL:
                result.append(f"{i}. Roll all dice (EV: {value:.1f})")
            elif action.kind == ActionType.HOLD:
                held = [i + 1 for i, hold in enumerate(action.data) if hold]
                if held:
                    held_values = [current_state.current_dice[i-1] for i in held]
                    result.append(f"{i}. Hold {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))} (EV: {value:.1f})")
                else:
                    result.append(f"{i}. Roll all dice (EV: {value:.1f})")
            else:
                points = env.calc_score(action.data, current_state.current_dice)
                result.append(f"{i}. Score {action.data.name} for {points} points (EV: {value:.1f})")
    
    # Format current state
    state_display = []
    if current_state is not None:
        state_display.append(format_dice_state(current_state.current_dice, current_state.rolls_left))
        state_display.append(format_score_sheet(current_state))
        state_display.append(format_combinations(current_state.current_dice))
    
    return "\n".join(result), "\n".join(state_display)

def run_performance_analysis(num_games: int = 100) -> Tuple[str, None]:
    """Run performance analysis and return statistics."""
    if agent is None:
        return "Please load a model first.", None
        
    try:
        # Run evaluation
        actual_scores = []
        training_rewards = []
        old_eps = agent.epsilon
        agent.epsilon = 0.02
        
        for _ in range(num_games):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state_vec = encoder.encode(state)
                valid_actions = env.get_valid_actions()
                
                if not valid_actions:
                    break
                    
                q_values = agent.get_q_values(state_vec)
                mask = np.full(agent.action_size, float("-inf"))
                mask[valid_actions] = 0
                q_values = q_values + mask
                
                valid_q = [(i, q_values[i]) for i in valid_actions]
                valid_q.sort(key=lambda x: x[1], reverse=True)
                action_idx = valid_q[0][0]
                
                state, reward, done, _ = env.step(action_idx)
                total_reward += reward
                
            training_rewards.append(total_reward)
            
            # Calculate actual game score
            upper_total = sum(state.score_sheet[cat] or 0 for cat in [
                YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
                YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES
            ])
            bonus = 35 if upper_total >= 63 else 0
            lower_total = sum(state.score_sheet[cat] or 0 for cat in [
                YahtzeeCategory.THREE_OF_A_KIND, YahtzeeCategory.FOUR_OF_A_KIND,
                YahtzeeCategory.FULL_HOUSE, YahtzeeCategory.SMALL_STRAIGHT,
                YahtzeeCategory.LARGE_STRAIGHT, YahtzeeCategory.YAHTZEE,
                YahtzeeCategory.CHANCE
            ])
            actual_score = upper_total + bonus + lower_total
            actual_scores.append(actual_score)
            
        agent.epsilon = old_eps
        
        # Calculate statistics
        actual_scores = np.array(actual_scores)
        training_rewards = np.array(training_rewards)
        
        # Format statistics with clear sections
        output = []
        
        # Header
        output.append(f"Performance Analysis ({num_games} games)")
        output.append("=" * 40)
        output.append("")
        
        # Game Score Statistics
        output.append("Game Score Statistics")
        output.append("-" * 20)
        output.append(f"Mean:    {np.mean(actual_scores):>6.1f} ± {np.std(actual_scores):.1f}")
        output.append(f"Median:  {np.median(actual_scores):>6.1f}")
        output.append(f"Range:   {np.min(actual_scores):>6.1f} - {np.max(actual_scores):.1f}")
        output.append("")
        
        # Score Distribution
        output.append("Score Distribution")
        output.append("-" * 20)
        output.append(f"{'Range':<10} {'Count':>6} {'%':>6}  {'Distribution':<30}")
        output.append("-" * 60)
        
        # Calculate distribution with visual bars
        brackets = [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, float('inf'))]
        max_count = max(np.sum((actual_scores >= low) & (actual_scores < high)) for low, high in brackets)
        bar_scale = 30.0 / max_count if max_count > 0 else 0
        
        for low, high in brackets:
            count = np.sum((actual_scores >= low) & (actual_scores < high))
            percentage = (count / num_games) * 100
            bar = "█" * int(count * bar_scale)
            high_str = str(high) if high != float('inf') else "inf"
            output.append(f"{low:>3}-{high_str:<6} {count:>6} {percentage:>5.1f}%  {bar}")
        
        # Join with proper newlines for Gradio display
        return "\n".join(output).replace("\n", "<br>"), None
        
    except Exception as e:
        return f"Error during performance analysis: {str(e)}", None

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(title="Yahtzee RL") as interface:
        gr.Markdown("""# Yahtzee RL
        Select a model and objective (win/avg_score), then load it to begin. Use the tabs below to:
        - Simulation: Watch a full game
        - Calculation: Analyze moves step by step
        - Analysis: Get performance statistics
        """)
        
        # Model and objective selection
        with gr.Row():
            model_choices = get_available_models()
            model_dropdown = gr.Dropdown(
                choices=[format_model_name(m) for m in model_choices],
                value=format_model_name(model_choices[0]) if model_choices else None,
                label="Select Model",
                info="Choose a trained model to use"
            )
            objective_radio = gr.Radio(
                choices=["win", "avg_score"],
                value="win",
                label="Objective",
                info="win: vs opponent, avg_score: maximize points"
            )
            load_button = gr.Button("Load Model")
            model_status = gr.Textbox(label="Status", interactive=False)
            
        # Create tabs for different modes
        with gr.Tabs() as tabs:
            # Simulation Mode
            with gr.Tab("Simulation"):
                gr.Markdown("Click 'Start New Game' to watch the AI play a complete game.")
                sim_button = gr.Button("Start New Game")
                sim_output = gr.Textbox(
                    label="Game Log",
                    interactive=False,
                    lines=25
                )
                sim_score = gr.Textbox(
                    label="Final Result",
                    interactive=False
                )
            
            # Calculation Mode
            with gr.Tab("Calculation"):
                gr.Markdown("1. Click 'New Game State' 2. Choose action number (1-5) 3. Click 'Take Action'")
                with gr.Row():
                    new_state_button = gr.Button("New Game State")
                    action_input = gr.Number(
                        label="Action Number",
                        value=1,
                        minimum=1,
                        maximum=5,
                        step=1
                    )
                    take_action_button = gr.Button("Take Action")
                with gr.Row():
                    action_output = gr.Textbox(
                        label="Action Analysis",
                        interactive=False,
                        lines=10
                    )
                    state_output = gr.Textbox(
                        label="Current State",
                        interactive=False,
                        lines=10
                    )
            
            # Performance Stats
            with gr.Tab("Analysis"):
                gr.Markdown("Adjust number of games and click 'Run Analysis' to see performance statistics.")
                num_games_input = gr.Slider(
                    minimum=10,
                    maximum=1000,
                    value=100,
                    step=10,
                    label="Number of Games"
                )
                stats_button = gr.Button("Run Analysis")
                stats_output = gr.HTML(
                    label="Results",
                    value=""
                )
        
        # Connect components
        def on_load_model(choice: str, objective: str) -> str:
            if not choice:
                return "Please select a model"
            idx = [format_model_name(m) for m in model_choices].index(choice)
            return load_model(model_choices[idx])
        
        load_button.click(on_load_model, inputs=[model_dropdown, objective_radio], outputs=[model_status])
        sim_button.click(run_simulation, outputs=[sim_output, sim_score])
        new_state_button.click(analyze_state, inputs=[gr.Textbox(value="new", visible=False)], outputs=[action_output, state_output])
        take_action_button.click(take_action, inputs=[action_input], outputs=[action_output, state_output])
        stats_button.click(run_performance_analysis, inputs=[num_games_input], outputs=[stats_output])
        
    return interface

# Create the demo interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 