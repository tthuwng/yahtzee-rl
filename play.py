import argparse
import os
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import IDX_TO_ACTION, NUM_ACTIONS, ActionType, GameState, YahtzeeEnv, YahtzeeCategory
from utils import visualize_game_state


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


def load_agent(model_path: Optional[str] = None) -> YahtzeeAgent:
    """Load a trained agent from a model file."""
    if model_path is None:
        model_path = select_model()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = StateEncoder()
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


def simulate_game(agent: YahtzeeAgent, delay: float = 0.5) -> float:
    """Run a full game simulation with visualization."""
    env = YahtzeeEnv()
    encoder = StateEncoder()
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    # Store original epsilon and set to minimum for deterministic play
    old_eps = agent.epsilon
    agent.epsilon = 0.02

    while not done:
        clear_output(wait=True)
        print(f"\n=== Turn {turn} | Score: {total_reward:.0f} ===")
        
        # Show game state
        visualize_game_state(state, env)

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
        
        # Get top 3 actions
        valid_q = [(i, q_values[i]) for i in valid_actions]
        valid_q.sort(key=lambda x: x[1], reverse=True)
        top_actions = valid_q[:3]
        
        # Show top actions
        print("\nTop Actions:")
        for i, (action_idx, value) in enumerate(top_actions, 1):
            action = IDX_TO_ACTION[action_idx]
            if action.kind == ActionType.ROLL:
                print(f"{i}. Roll all dice (EV: {value:.1f})")
            elif action.kind == ActionType.HOLD:
                held = [i + 1 for i, hold in enumerate(action.data) if hold]
                if held:
                    held_values = [state.current_dice[i-1] for i in held]
                    print(f"{i}. Hold {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))} (EV: {value:.1f})")
            else:
                score = env.calc_score(action.data, state.current_dice)
                print(f"{i}. Score {action.data.name} for {score} points (EV: {value:.1f})")
        
        # Take best action
        action_idx = top_actions[0][0]
        action = IDX_TO_ACTION[action_idx]
        
        # Show decision
        print("\nAgent's Decision:")
        if action.kind == ActionType.ROLL:
            print("Rolling all dice")
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                held_values = [state.current_dice[i-1] for i in held]
                print(f"Holding: {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))}")
            else:
                print("Rolling all dice")
        else:
            score = env.calc_score(action.data, state.current_dice)
            print(f"Scoring {action.data.name} for {score} points")

        # Take action
        state, reward, done, _ = env.step(action_idx)
        total_reward += reward
        
        if action.kind == ActionType.SCORE:
            turn += 1
            
        time.sleep(delay)

    # Show final results
    clear_output(wait=True)
    print("\n=== Game Over ===")
    visualize_game_state(state, env)
    
    # Calculate final scores
    upper_score = sum(state.score_sheet[cat] or 0 for cat in [
        YahtzeeCategory.ONES, YahtzeeCategory.TWOS, YahtzeeCategory.THREES,
        YahtzeeCategory.FOURS, YahtzeeCategory.FIVES, YahtzeeCategory.SIXES
    ])
    bonus = env.calc_upper_bonus()
    lower_score = total_reward - upper_score - bonus
    
    print(f"\nFinal Score: {total_reward:.0f}")
    print(f"• Upper Section: {upper_score}")
    print(f"• Upper Bonus: {bonus}")
    print(f"• Lower Section: {lower_score:.0f}")

    agent.epsilon = old_eps
    return total_reward


def show_action_values(
    agent: YahtzeeAgent, state: Optional[GameState] = None, num_top: int = 5
) -> Tuple[GameState, list]:
    """Show expected values for all valid actions in current state."""
    env = YahtzeeEnv()
    encoder = StateEncoder()

    if state is None:
        state = env.reset()

    print("\nCurrent Game State:")
    visualize_game_state(state, env)

    # Get state encoding and valid actions
    state_vec = encoder.encode(state)
    valid_actions = env.get_valid_actions()

    # Get Q-values and mask invalid actions
    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask

    # Sort actions by Q-value
    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Actions and Their Expected Values:")
    for i, (action_idx, value) in enumerate(valid_q[:num_top]):
        action = env.IDX_TO_ACTION[action_idx]
        if action.kind == ActionType.ROLL:
            print(f"{i+1}. ROLL all dice (EV: {value:.1f})")
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                print(f"{i+1}. Hold dice {held} (EV: {value:.1f})")
            else:
                print(f"{i+1}. ROLL all dice (EV: {value:.1f})")
        else:
            print(f"{i+1}. Score {action.data.name} (EV: {value:.1f})")

    return state, valid_q[:num_top]


def evaluate_performance(agent: YahtzeeAgent, num_games: int = 100) -> None:
    """Run multiple games and show performance statistics."""
    print(f"\nEvaluating agent performance over {num_games} games...")
    scores = []
    env = YahtzeeEnv()
    encoder = StateEncoder()

    # Store original epsilon and set to minimum for deterministic evaluation
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
            print(f"Completed {i + 1} games...")

    # Restore original epsilon
    agent.epsilon = old_eps

    # Calculate statistics
    scores = np.array(scores)
    print("\nPerformance Statistics:")
    print(f"Mean Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Median Score: {np.median(scores):.1f}")
    print(f"Min Score: {np.min(scores):.1f}")
    print(f"Max Score: {np.max(scores):.1f}")
    
    # Plot score distribution
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Play Yahtzee with a trained agent")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the trained model file (optional, will prompt for selection if not provided)",
    )
    args = parser.parse_args()

    # Load the agent
    agent = load_agent(args.model)

    while True:
        print("\nYahtzee AI Interface")
        print("===================")
        print("1. Watch Agent Play (Simulation Mode)")
        print("2. Analyze Moves (Calculation Mode)")
        print("3. Evaluate Agent Performance")
        print("4. Load Different Model")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            print("\nStarting simulation mode...")
            simulate_game(agent)
            input("\nPress Enter to continue...")

        elif choice == "2":
            print("\nStarting calculation mode...")
            current_state = None

            while True:
                current_state, valid_actions = show_action_values(agent, current_state)

                print("\nOptions:")
                print("1. Take an action and continue")
                print("2. Start new game")
                print("3. Return to main menu")

                subchoice = input("\nEnter choice (1-3): ").strip()

                if subchoice == "1":
                    try:
                        action_num = int(input("\nEnter action number: ").strip())
                        if 1 <= action_num <= len(valid_actions):
                            action_idx = valid_actions[action_num - 1][0]
                            env = YahtzeeEnv()
                            current_state, reward, done, _ = env.step(action_idx)

                            if done:
                                print(f"\nGame Over! Final Score: {reward:.1f}")
                                break
                        else:
                            print("\nInvalid action number!")
                    except ValueError:
                        print("\nPlease enter a valid number!")

                elif subchoice == "2":
                    current_state = None
                else:
                    break

        elif choice == "3":
            num_games = int(input("\nEnter number of games to evaluate (default 100): ") or "100")
            evaluate_performance(agent, num_games)
            input("\nPress Enter to continue...")

        elif choice == "4":
            agent = load_agent()  # Will prompt for model selection
            print("\nNew model loaded successfully!")

        elif choice == "5":
            print("\nThanks for playing!")
            break

        else:
            print("\nInvalid choice! Please try again.")


if __name__ == "__main__":
    main()
