import argparse
import time
from typing import Optional, Tuple

import numpy as np
import torch
from IPython.display import clear_output

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import IDX_TO_ACTION, NUM_ACTIONS, ActionType, GameState, YahtzeeEnv
from utils import visualize_game_state


def load_agent(model_path: str) -> YahtzeeAgent:
    """Load a trained agent from a model file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = StateEncoder()
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=NUM_ACTIONS,
        batch_size=1024,
        gamma=0.99,
        learning_rate=2e-4,
        target_update=250,
        use_boltzmann=True,
        device=device,
    )

    # Load trained weights
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    return agent


def simulate_game(agent: YahtzeeAgent, delay: float = 1.0) -> float:
    """Run a full game simulation with visualization."""
    env = YahtzeeEnv()
    encoder = StateEncoder()
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    while not done:
        clear_output(wait=True)
        print(f"\n=== Turn {turn} ===")
        visualize_game_state(state, env)

        state_vec = encoder.encode(state)
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        # Get Q-values and select best action
        q_values = agent.get_q_values(state_vec)
        mask = np.full(agent.action_size, float("-inf"))
        mask[valid_actions] = 0
        q_values = q_values + mask
        action_idx = q_values.argmax()
        action = IDX_TO_ACTION[action_idx]

        # Display agent's decision
        print("\nAgent's decision:")
        if action.kind == ActionType.ROLL:
            print("Action: ROLL all dice")
        elif action.kind == ActionType.HOLD_MASK:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            print(f"Action: Hold dice at positions {held}")
        else:
            print(f"Action: Score {action.data.name}")
            print(f"Expected value: {q_values[action_idx]:.1f}")
            turn += 1

        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        if action.kind == ActionType.SCORE:
            print(f"Scored {reward:.1f} points")

        time.sleep(delay)

    clear_output(wait=True)
    print("\n=== Game Over ===")
    visualize_game_state(state, env)
    print(f"\nFinal Score: {total_reward:.1f}")

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
        elif action.kind == ActionType.HOLD_MASK:
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

    for i in range(num_games):
        score = simulate_game(agent, delay=0)  # No delay for evaluation
        scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} games...")

    scores = np.array(scores)
    print("\nPerformance Statistics:")
    print(f"Mean Score: {np.mean(scores):.1f}")
    print(f"Median Score: {np.median(scores):.1f}")
    print(f"Standard Deviation: {np.std(scores):.1f}")
    print(f"Min Score: {np.min(scores):.1f}")
    print(f"Max Score: {np.max(scores):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Play Yahtzee with a trained agent")
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.pth",
        help="Path to the trained model file",
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
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

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
            evaluate_performance(agent)
            input("\nPress Enter to continue...")

        elif choice == "4":
            print("\nThanks for playing!")
            break

        else:
            print("\nInvalid choice! Please try again.")


if __name__ == "__main__":
    main()
