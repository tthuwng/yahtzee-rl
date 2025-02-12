import argparse
import os
import time
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm

from dqn import YahtzeeAgent
from encoder import IDX_TO_ACTION, NUM_ACTIONS, StateEncoder
from env import ActionType, GameState, YahtzeeEnv, YahtzeeCategory, RewardStrategy


def list_available_models(models_dir: str = "models") -> List[str]:
    """List all available model files in the models directory."""
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith('.pth')]


def select_model() -> str:
    """Interactive model selection from available models."""
    models = list_available_models()
    if not models:
        print("No models found in the models directory!")
        exit(1)
        
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
        
    while True:
        try:
            choice = int(input("\nSelect model number: "))
            if 1 <= choice <= len(models):
                return os.path.join("models", models[choice - 1])
            print("Invalid choice! Please try again.")
        except ValueError:
            print("Please enter a valid number!")


def convert_legacy_model(legacy_state: dict, state_size: int, action_size: int, device: torch.device) -> dict:
    """Convert legacy model state dict to new architecture format."""
    new_state = {}
    
    # Convert feature layers to new architecture
    if "features.0.weight" in legacy_state:
        # Move all legacy tensors to device first
        legacy_state = {k: v.to(device) for k, v in legacy_state.items()}
        
        # Input layer
        new_state["input_layer.0.weight"] = legacy_state["features.0.weight"]
        new_state["input_layer.0.bias"] = legacy_state["features.0.bias"]
        new_state["input_layer.2.weight"] = legacy_state["features.2.weight"]
        new_state["input_layer.2.bias"] = legacy_state["features.2.bias"]
        
        # Initialize residual blocks
        for block_num, prefix in [(1, "res_block1"), (2, "res_block2")]:
            # First layer of residual block
            new_state[f"{prefix}.0.weight"] = legacy_state["features.3.weight"].clone()
            new_state[f"{prefix}.0.bias"] = legacy_state["features.3.bias"].clone()
            new_state[f"{prefix}.2.weight"] = legacy_state["features.5.weight"].clone()
            new_state[f"{prefix}.2.bias"] = legacy_state["features.5.bias"].clone()
            
            # Second layer with size adjustment
            old_weight = legacy_state["features.6.weight"]  # [256, 512]
            old_bias = legacy_state["features.6.bias"]      # [256]
            
            # Expand to [512, 512] by repeating
            new_state[f"{prefix}.3.weight"] = torch.cat([old_weight, old_weight], dim=0)
            new_state[f"{prefix}.3.bias"] = torch.cat([old_bias, old_bias], dim=0)
            
            # Layer norm weights/bias
            old_weight = legacy_state["features.8.weight"]  # [256]
            old_bias = legacy_state["features.8.bias"]      # [256]
            
            # Expand to [512] by repeating
            new_state[f"{prefix}.5.weight"] = torch.cat([old_weight, old_weight], dim=0)
            new_state[f"{prefix}.5.bias"] = torch.cat([old_bias, old_bias], dim=0)
        
        # Output layer
        new_state["output_layer.0.weight"] = legacy_state["features.6.weight"].clone()
        new_state["output_layer.0.bias"] = legacy_state["features.6.bias"].clone()
        new_state["output_layer.2.weight"] = legacy_state["features.8.weight"].clone()
        new_state["output_layer.2.bias"] = legacy_state["features.8.bias"].clone()
        
        # Value stream
        old_value_weight = legacy_state["value_stream.0.weight"]  # [128, 256]
        old_value_bias = legacy_state["value_stream.0.bias"]      # [128]
        
        # Expand to new sizes
        new_state["value_stream.0.weight"] = torch.cat([old_value_weight, old_value_weight], dim=0)
        new_state["value_stream.0.bias"] = torch.cat([old_value_bias, old_value_bias], dim=0)
        new_state["value_stream.2.weight"] = torch.ones(256, dtype=torch.float32, device=device) * legacy_state["value_stream.2.weight"].mean()
        new_state["value_stream.2.bias"] = torch.ones(256, dtype=torch.float32, device=device) * legacy_state["value_stream.2.bias"].mean()
        
        # Final value layer with size adjustment
        old_weight = legacy_state["value_stream.2.weight"]  # [1, 128]
        # Expand to [1, 256] by repeating
        new_state["value_stream.3.weight"] = torch.cat([old_weight, old_weight], dim=1)
        new_state["value_stream.3.bias"] = legacy_state["value_stream.2.bias"].clone()
        
        # Advantage stream
        old_adv_weight = legacy_state["advantage_stream.0.weight"]  # [128, 256]
        old_adv_bias = legacy_state["advantage_stream.0.bias"]      # [128]
        
        # Expand to new sizes
        new_state["advantage_stream.0.weight"] = torch.cat([old_adv_weight, old_adv_weight], dim=0)
        new_state["advantage_stream.0.bias"] = torch.cat([old_adv_bias, old_adv_bias], dim=0)
        new_state["advantage_stream.2.weight"] = torch.ones(256, dtype=torch.float32, device=device) * legacy_state["advantage_stream.2.weight"].mean()
        new_state["advantage_stream.2.bias"] = torch.ones(256, dtype=torch.float32, device=device) * legacy_state["advantage_stream.2.bias"].mean()
        
        # Final advantage layer with size adjustment
        old_weight = legacy_state["advantage_stream.2.weight"]  # [46, 128]
        # Expand to [46, 256] by repeating
        new_state["advantage_stream.3.weight"] = torch.cat([old_weight, old_weight], dim=1)
        new_state["advantage_stream.3.bias"] = legacy_state["advantage_stream.2.bias"].clone()
        
        return new_state
    else:
        raise Exception("Unrecognized model format")


def load_agent(model_path: Optional[str] = None, objective: str = "win") -> YahtzeeAgent:
    """Load a trained agent from a model file."""
    if model_path is None:
        model_path = select_model()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize encoder based on objective
    encoder = StateEncoder(use_opponent_value=(objective == "win"))
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=NUM_ACTIONS,
        batch_size=512,
        gamma=0.997,
        learning_rate=5e-5,
        target_update=50,
        device=device,
    )

    # Load model using the agent's load method
    agent.load(model_path)

    # Set to eval mode
    agent.eval()
    return agent


def simulate_game(agent: YahtzeeAgent, delay: float = 0.5) -> Tuple[float, float]:
    """
    Run a full game simulation with visualization.
    Returns (actual_score, training_reward)
    """
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=True)  # Set to True to match training
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    # Store original epsilon and set to minimum for deterministic play
    old_eps = agent.epsilon
    agent.epsilon = 0.02  # Match training evaluation epsilon

    while not done:
        # Calculate actual game score for display
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
        
        clear_output(wait=True)
        print(f"\n=== Turn {turn} | Score: {actual_score} (Reward: {total_reward:.1f}) ===")
        print(env.render())

        state_vec = encoder.encode(state, opponent_value=0.5)  # Add opponent_value=0.5
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        action_idx = agent.select_action(state_vec, valid_actions)
        action = IDX_TO_ACTION[action_idx]

        print("\nAgent's decision:")
        if action.kind == ActionType.ROLL:
            print("Action: ROLL all dice")
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                held_values = [state.current_dice[i-1] for i in held]
                print(f"Action: Hold {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))}")
            else:
                print("Action: ROLL all dice")
        else:
            points = env.calc_score(action.data, state.current_dice)
            print(f"Action: Score {action.data.name} for {points} points")

        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        if action.kind == ActionType.SCORE:
            points = env.calc_score(action.data, state.current_dice)
            print(f"Scored {points} points (Reward: {reward:.1f})")
            turn += 1
        time.sleep(delay)

    # Calculate final actual score
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

    clear_output(wait=True)
    print("\n=== Game Over ===")
    print(env.render())
    print(f"\nFinal Results:")
    print(f"• Actual Score: {actual_score}")
    print(f"  - Upper Section: {upper_total}")
    print(f"  - Upper Bonus: {bonus}")
    print(f"  - Lower Section: {lower_total}")
    print(f"• Training Reward: {total_reward:.1f}")

    agent.epsilon = old_eps
    return actual_score, total_reward


def show_action_values(
    agent: YahtzeeAgent, state: Optional[GameState] = None, num_top: int = 5
) -> tuple:
    """
    Show expected values for all valid actions in the current state.
    If state is None, starts a new game.
    Returns (state, valid_actions, q_values) for further use.
    """
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=True)  # Set to True to match training

    if state is None:
        state = env.reset()

    print("\nCurrent Game State:")
    print(env.render())

    state_vec = encoder.encode(state, opponent_value=0.5)  # Add opponent_value=0.5
    valid_actions = env.get_valid_actions()

    # Show dice state
    dice_values = state.current_dice
    dice_str = " ".join(f"[{d}]" if d > 0 else "[ ]" for d in dice_values)
    print(f"\nDice (Rolls Left: {state.rolls_left})")
    print(f"Positions: [1] [2] [3] [4] [5]")
    print(f"Values:   {dice_str}")
    
    # Show score board
    print("\nScore Board:")
    print("─" * 20)
    print(f"{'Category':<12} Score")
    print("─" * 20)
    
    # Upper section
    upper_total = 0
    for cat in [YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
               YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES]:
        score = state.score_sheet[cat]
        upper_total += score if score is not None else 0
        print(f"{cat.name:<12} {score if score is not None else '-':>5}")
    
    # Calculate upper section bonus
    bonus = 35 if upper_total >= 63 else 0
    print("─" * 20)
    print(f"{'Sum':<12} {upper_total:>5}")
    print(f"{'Bonus':<12} {bonus:>5}")
    print("─" * 20)
    
    # Lower section
    lower_total = 0
    for cat in [YahtzeeCategory.THREE_OF_A_KIND, YahtzeeCategory.FOUR_OF_A_KIND,
               YahtzeeCategory.FULL_HOUSE, YahtzeeCategory.SMALL_STRAIGHT,
               YahtzeeCategory.LARGE_STRAIGHT, YahtzeeCategory.YAHTZEE,
               YahtzeeCategory.CHANCE]:
        score = state.score_sheet[cat]
        lower_total += score if score is not None else 0
        print(f"{cat.name:<12} {score if score is not None else '-':>5}")
    
    # Show total
    total = upper_total + bonus + lower_total
    print("─" * 20)
    print(f"{'TOTAL':<12} {total:>5}")
    
    # Show current combinations if any dice showing
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

    # Get Q-values and mask invalid actions
    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask

    # Sort actions by Q-value
    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Actions and Their Expected Values:")
    for i, (action_idx, value) in enumerate(valid_q[:num_top], 1):
        action = env.IDX_TO_ACTION[action_idx]
        if action.kind == ActionType.ROLL:
            print(f"{i}. Roll all dice (EV: {value:.1f})")
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                held_values = [state.current_dice[i-1] for i in held]
                print(f"{i}. Hold {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))} (EV: {value:.1f})")
            else:
                print(f"{i}. Roll all dice (EV: {value:.1f})")
        else:
            points = env.calc_score(action.data, state.current_dice)
            print(f"{i}. Score {action.data.name} for {points} points (EV: {value:.1f})")

    return state, valid_q[:num_top], q_values


def evaluate_performance(agent: YahtzeeAgent, num_games: int = 50) -> None:
    """Run multiple games and show detailed performance statistics."""
    print(f"\nEvaluating agent performance over {num_games} games...")
    scores = []
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=True)

    # Store original epsilon and set to minimum for deterministic evaluation
    old_eps = agent.epsilon
    agent.epsilon = 0.02

    # Run games with progress bar
    for i in tqdm(range(num_games), desc="Playing games"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_vec = encoder.encode(state, opponent_value=0.5)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action_idx = agent.select_action(state_vec, valid_actions)
            state, reward, done, _ = env.step(action_idx)
            total_reward += reward

        scores.append(total_reward)

    # Restore original epsilon
    agent.epsilon = old_eps

    # Calculate detailed statistics
    scores = np.array(scores)
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    percentile_25 = np.percentile(scores, 25)
    percentile_75 = np.percentile(scores, 75)

    print("\n=== Performance Statistics ===")
    print(f"Number of games: {num_games}")
    print("\nKey Metrics:")
    print(f"• Mean Score:   {mean_score:.1f} ± {std_score:.1f}")
    print(f"• Median Score: {median_score:.1f}")
    print(f"• Min Score:    {min_score:.1f}")
    print(f"• Max Score:    {max_score:.1f}")
    print("\nScore Distribution:")
    print(f"• 25th percentile: {percentile_25:.1f}")
    print(f"• 75th percentile: {percentile_75:.1f}")
    print(f"• IQR:            {(percentile_75 - percentile_25):.1f}")
    
    # Calculate score brackets
    brackets = [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, float('inf'))]
    print("\nScore Distribution:")
    for low, high in brackets:
        count = np.sum((scores >= low) & (scores < high))
        percentage = (count / num_games) * 100
        high_str = f"{high:.0f}" if high != float('inf') else "inf"
        print(f"• {low:3.0f}-{high_str:>3}: {count:3.0f} games ({percentage:4.1f}%)")
    
    # Plot score distribution
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.1f}')
    plt.axvline(median_score, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_score:.1f}')
    plt.xlabel('Score')
    plt.ylabel('Number of Games')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(scores, vert=False)
    plt.xlabel('Score')
    plt.title('Score Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Yahtzee RL Challenge - Deep Q-Learning Agent")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the trained model file (optional, will prompt for selection if not provided)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games to play in performance evaluation mode (default: 50)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["win", "avg_score"],
        default="win",
        help="Evaluation objective (win or avg_score)",
    )
    args = parser.parse_args()

    # Load the agent
    agent = load_agent(args.model, args.objective)
    current_objective = args.objective

    while True:
        print("\n=== Yahtzee RL Challenge ===")
        print("\nCurrent Settings:")
        print(f"• Objective: {current_objective.upper()}")
        print(f"• Games per evaluation: {args.num_games}")
        print("\nSelect a mode:")
        print("• [1] Simulation Mode - Watch agent play a full game")
        print("• [2] Calculation Mode - Analyze expected values for each action")
        print(f"• [3] Performance Stats - View agent's statistics over {args.num_games} games")
        print("• [4] Load Different Model")
        print("• [5] Change Objective")
        print("• [6] Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            print("\n=== Simulation Mode ===")
            print("Watch the agent play a full game of Yahtzee.")
            print("The agent will show its decision-making process for each move.")
            input("Press Enter to start the game...")
            simulate_game(agent)
            
            print("\nWould you like to:")
            print("1. Play another game")
            print("2. Return to main menu")
            subchoice = input("\nEnter choice (1-2): ").strip()
            if subchoice != "1":
                continue

        elif choice == "2":
            print("\n=== Calculation Mode ===")
            print("Analyze the expected value of each possible action in any game state.")
            current_state = None

            while True:
                current_state, valid_actions, q_values = show_action_values(agent, current_state)

                print("\nOptions:")
                print("• [1] Take an action and continue analysis")
                print("• [2] Start new game state")
                print("• [3] Return to main menu")

                subchoice = input("\nEnter choice (1-3): ").strip()

                if subchoice == "1":
                    try:
                        action_num = int(input("\nEnter action number to simulate: ").strip())
                        if 1 <= action_num <= len(valid_actions):
                            action_idx = valid_actions[action_num - 1][0]
                            env = YahtzeeEnv()
                            current_state, reward, done, _ = env.step(action_idx)

                            if done:
                                print(f"\n=== Game Over! Final Score: {reward:.0f} ===")
                                break
                        else:
                            print("\nInvalid action number! Please try again.")
                    except ValueError:
                        print("\nPlease enter a valid number!")

                elif subchoice == "2":
                    current_state = None
                else:
                    break

        elif choice == "3":
            print("\n=== Performance Statistics ===")
            print(f"Running {args.num_games} full games to evaluate agent performance...")
            evaluate_performance(agent, args.num_games)
            input("\nPress Enter to return to menu...")

        elif choice == "4":
            print("\n=== Load New Model ===")
            agent = load_agent(objective=current_objective)  # Will prompt for model selection
            print("\nNew model loaded successfully!")

        elif choice == "5":
            print("\n=== Change Objective ===")
            print("Select evaluation objective:")
            print("1. Win Mode (considers opponent strength)")
            print("2. Average Score Mode (pure score maximization)")
            
            while True:
                subchoice = input("\nEnter choice (1-2): ").strip()
                if subchoice == "1":
                    current_objective = "win"
                    break
                elif subchoice == "2":
                    current_objective = "avg_score"
                    break
                else:
                    print("Invalid choice! Please try again.")
            
            # Reload agent with new objective
            agent = load_agent(args.model, current_objective)
            print(f"\nSwitched to {current_objective.upper()} mode")

        elif choice == "6":
            print("\nThank you for using the Yahtzee RL Challenge demo!")
            break

        else:
            print("\nInvalid choice! Please try again.")


if __name__ == "__main__":
    main()
