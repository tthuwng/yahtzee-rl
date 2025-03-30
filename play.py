import argparse
import os
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import GameState, RewardStrategy, YahtzeeEnv
from yahtzee_types import ActionType


def list_available_models(models_dir: str = "models") -> List[str]:
    """List all available model files in the models directory."""
    if not os.path.exists(models_dir):
        return []

    model_files = []
    # Look for .pth files in models directory and its subdirectories
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pth"):
                model_files.append(os.path.join(root, file))

    return model_files


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
            choice = int(input("\nSelect a model number: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print(
                    f"Please enter a number between 1 and {len(models)}"
                )
        except ValueError:
            print("Please enter a valid number")


def convert_legacy_model(
    legacy_state: dict,
    state_size: int,
    action_size: int,
    device: torch.device,
) -> dict:
    """Convert legacy model state dict to new architecture format."""
    new_state = {}

    # Check if this is already a new format
    if "features.0.weight" not in legacy_state:
        return legacy_state

    try:
        # Input layer
        new_state["input_layer.0.weight"] = legacy_state[
            "features.0.weight"
        ].clone()
        new_state["input_layer.0.bias"] = legacy_state[
            "features.0.bias"
        ].clone()
        new_state["input_layer.2.weight"] = legacy_state[
            "features.2.weight"
        ].clone()
        new_state["input_layer.2.bias"] = legacy_state[
            "features.2.bias"
        ].clone()

        for block_num, prefix in [
            (1, "res_block1"),
            (2, "res_block2"),
        ]:
            # First layer of residual block
            new_state[f"{prefix}.0.weight"] = legacy_state[
                "features.3.weight"
            ].clone()
            new_state[f"{prefix}.0.bias"] = legacy_state[
                "features.3.bias"
            ].clone()
            new_state[f"{prefix}.2.weight"] = legacy_state[
                "features.5.weight"
            ].clone()
            new_state[f"{prefix}.2.bias"] = legacy_state[
                "features.5.bias"
            ].clone()

            # Second layer with size adjustment
            old_weight = legacy_state[
                "features.6.weight"
            ]  # [256, 512]
            old_bias = legacy_state["features.6.bias"]  # [256]

            # Expand to [512, 512] by repeating
            new_state[f"{prefix}.3.weight"] = torch.cat(
                [old_weight, old_weight], dim=0
            )
            new_state[f"{prefix}.3.bias"] = torch.cat(
                [old_bias, old_bias], dim=0
            )

            # Layer norm weights/bias
            old_weight = legacy_state["features.8.weight"]  # [256]
            old_bias = legacy_state["features.8.bias"]  # [256]

            # Expand to [512] by repeating
            new_state[f"{prefix}.5.weight"] = torch.cat(
                [old_weight, old_weight], dim=0
            )
            new_state[f"{prefix}.5.bias"] = torch.cat(
                [old_bias, old_bias], dim=0
            )

        # Output layer
        new_state["output_layer.0.weight"] = legacy_state[
            "features.6.weight"
        ].clone()
        new_state["output_layer.0.bias"] = legacy_state[
            "features.6.bias"
        ].clone()
        new_state["output_layer.2.weight"] = legacy_state[
            "features.8.weight"
        ].clone()
        new_state["output_layer.2.bias"] = legacy_state[
            "features.8.bias"
        ].clone()

        # Value stream
        old_value_weight = legacy_state[
            "value_stream.0.weight"
        ]  # [128, 256]
        old_value_bias = legacy_state["value_stream.0.bias"]  # [128]

        # Expand to new sizes
        new_state["value_stream.0.weight"] = torch.cat(
            [old_value_weight, old_value_weight], dim=0
        )
        new_state["value_stream.0.bias"] = torch.cat(
            [old_value_bias, old_value_bias], dim=0
        )
        new_state["value_stream.2.weight"] = (
            torch.ones(256, dtype=torch.float32, device=device)
            * legacy_state["value_stream.2.weight"].mean()
        )
        new_state["value_stream.2.bias"] = (
            torch.ones(256, dtype=torch.float32, device=device)
            * legacy_state["value_stream.2.bias"].mean()
        )

        # Final value layer with size adjustment
        old_weight = legacy_state["value_stream.2.weight"]  # [1, 128]
        # Expand to [1, 256] by repeating
        new_state["value_stream.3.weight"] = torch.cat(
            [old_weight, old_weight], dim=1
        )
        new_state["value_stream.3.bias"] = legacy_state[
            "value_stream.2.bias"
        ].clone()

        # Advantage stream
        old_adv_weight = legacy_state[
            "advantage_stream.0.weight"
        ]  # [128, 256]
        old_adv_bias = legacy_state[
            "advantage_stream.0.bias"
        ]  # [128]

        # Expand to new sizes
        new_state["advantage_stream.0.weight"] = torch.cat(
            [old_adv_weight, old_adv_weight], dim=0
        )
        new_state["advantage_stream.0.bias"] = torch.cat(
            [old_adv_bias, old_adv_bias], dim=0
        )
        new_state["advantage_stream.2.weight"] = (
            torch.ones(256, dtype=torch.float32, device=device)
            * legacy_state["advantage_stream.2.weight"].mean()
        )
        new_state["advantage_stream.2.bias"] = (
            torch.ones(256, dtype=torch.float32, device=device)
            * legacy_state["advantage_stream.2.bias"].mean()
        )

        # Final advantage layer with size adjustment
        old_weight = legacy_state[
            "advantage_stream.2.weight"
        ]  # [46, 128]
        # Expand to [46, 256] by repeating
        new_state["advantage_stream.3.weight"] = torch.cat(
            [old_weight, old_weight], dim=1
        )
        new_state["advantage_stream.3.bias"] = legacy_state[
            "advantage_stream.2.bias"
        ].clone()

        return new_state
    except Exception as e:
        print(f"Error converting legacy model: {e}")
        return legacy_state


def load_agent(
    model_path: Optional[str] = None, objective: str = "win"
) -> YahtzeeAgent:
    """Load a trained agent from a model file."""
    if model_path is None:
        model_path = select_model()

    print(f"Loading model from {model_path}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create encoder first to get state size
    encoder = StateEncoder(use_opponent_value=False)

    # Initialize agent with appropriate parameters
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=46,  # Fixed action space size
        batch_size=2048,
        gamma=0.997,
        learning_rate=1e-4,
        target_update=50,
        device=str(device),
    )

    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # If checkpoint is a dictionary with keys, use the new format
        if (
            isinstance(checkpoint, dict)
            and "policy_net" in checkpoint
        ):
            agent.policy_net.load_state_dict(checkpoint["policy_net"])
            agent.target_net.load_state_dict(checkpoint["target_net"])
            if "optimizer" in checkpoint:
                agent.optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )
            if "epsilon" in checkpoint:
                agent.epsilon = checkpoint["epsilon"]
            if "learn_steps" in checkpoint:
                agent.learn_steps = checkpoint["learn_steps"]
            print("Loaded model in new checkpoint format")
        else:
            # Try to load as legacy format
            converted_state = convert_legacy_model(
                checkpoint, encoder.state_size, 46, device
            )
            agent.policy_net.load_state_dict(converted_state)
            agent.target_net.load_state_dict(converted_state)
            print("Loaded and converted legacy model format")

        # Put the agent in evaluation mode
        agent.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    return agent


def simulate_game(agent: YahtzeeAgent, delay: float = 0.5) -> float:
    """
    Run a full game simulation with visualization.
    Returns the final score.
    """
    env = YahtzeeEnv(reward_strategy=RewardStrategy.STRATEGIC)
    encoder = StateEncoder(
        use_opponent_value=False
    )  # Set to match training
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    # Store original epsilon and set to minimum for deterministic play
    old_eps = agent.epsilon
    agent.epsilon = 0.02

    while not done:
        # Calculate current score
        upper_total = 0
        bonus = 0
        lower_total = 0

        for i, (cat, score) in enumerate(state.score_sheet.items()):
            if i < 6 and score is not None:  # Upper section
                upper_total += score
            elif score is not None:  # Lower section
                lower_total += score

        if upper_total >= 63:
            bonus = 35

        actual_score = upper_total + bonus + lower_total

        # Print current state
        print(f"\n=== Turn {turn} | Score: {actual_score} ===")
        print(env.render())

        # Get agent's action
        state_vec = encoder.encode(state)
        valid_actions = env.get_valid_actions()
        action_idx = agent.select_action(state_vec, valid_actions)
        action = env.action_mapper.index_to_action(action_idx)

        # Print agent's decision
        print("\nAgent's decision:")
        if action.kind == ActionType.ROLL:
            print("Action: ROLL all dice")
        elif action.kind == ActionType.HOLD:
            held = [
                i + 1 for i, hold in enumerate(action.data) if hold
            ]
            if held:
                held_values = [
                    state.current_dice[i - 1] for i in held
                ]
                print(
                    f"Action: Hold {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))}"
                )
            else:
                print("Action: ROLL all dice")
        else:
            points = env.calc_score(action.data, state.current_dice)
            print(
                f"Action: Score {action.data.name} for {points} points"
            )

        # Execute the action
        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        # Increment turn if scored
        if action.kind == ActionType.SCORE:
            turn += 1

        # Add delay for visualization
        if delay > 0:
            time.sleep(delay)

    # Calculate final score
    upper_total = 0
    bonus = 0
    lower_total = 0

    for i, (cat, score) in enumerate(state.score_sheet.items()):
        if i < 6 and score is not None:  # Upper section
            upper_total += score
        elif score is not None:  # Lower section
            lower_total += score

    if upper_total >= 63:
        bonus = 35

    actual_score = upper_total + bonus + lower_total

    # Print final state
    print("\n=== Game Over ===")
    print(env.render())
    print("\nFinal Results:")
    print(f"• Actual Score: {actual_score}")
    print(f"  - Upper Section: {upper_total}")
    print(f"  - Upper Bonus: {bonus}")
    print(f"  - Lower Section: {lower_total}")

    # Restore original epsilon
    agent.epsilon = old_eps

    return actual_score


def show_action_values(
    agent: YahtzeeAgent,
    state: Optional[GameState] = None,
    num_top: int = 5,
) -> tuple:
    """
    Show the agent's expected values for all valid actions in the current state.
    Returns (state, top_actions, q_values)
    """
    env = YahtzeeEnv()
    encoder = StateEncoder(
        use_opponent_value=True
    )  # Set to match training

    if state is None:
        state = env.reset()

    # Display current game state
    print("\nCurrent Game State:")
    print(env.render())

    # Get encoded state and all valid actions
    state_vec = encoder.encode(state, opponent_value=0.5)
    valid_actions = env.get_valid_actions()

    # Show dice state
    dice_values = state.current_dice
    dice_str = " ".join(
        f"[{d}]" if d > 0 else "[ ]" for d in dice_values
    )
    print(f"\nDice (Rolls Left: {state.rolls_left})")
    print("Positions: [1] [2] [3] [4] [5]")
    print(f"Values:   {dice_str}")

    # Show score board
    print("\nScoreboard:")
    print("-" * 20)
    print("Category      Score")
    print("-" * 20)

    # Upper section
    upper_total = 0
    for i, (cat, score) in enumerate(state.score_sheet.items()):
        if i < 6:  # Upper section
            score_val = score if score is not None else 0
            upper_total += score_val
            print(
                f"{cat.name:<12} {score if score is not None else '-':>5}"
            )

    # Show upper bonus
    bonus = 35 if upper_total >= 63 else 0
    bonus_needed = max(0, 63 - upper_total)
    print("-" * 20)
    print(f"{'Upper Total':<12} {upper_total:>5}")
    print(
        f"{'Bonus':<12} {bonus if upper_total >= 63 else f'Need {bonus_needed}':>5}"
    )
    print("-" * 20)

    # Lower section
    lower_total = 0
    for i, (cat, score) in enumerate(state.score_sheet.items()):
        if i >= 6:  # Lower section
            score_val = score if score is not None else 0
            lower_total += score_val
            print(
                f"{cat.name:<12} {score if score is not None else '-':>5}"
            )

    print("-" * 20)
    print(f"{'Lower Total':<12} {lower_total:>5}")
    print("-" * 20)
    print(
        f"{'Grand Total':<12} {upper_total + bonus + lower_total:>5}"
    )
    print("-" * 20)

    # Show dice combinations for reference
    if any(state.current_dice):
        dice_values = state.current_dice
        combinations = []
        counts = (
            np.bincount(dice_values)[1:] if any(dice_values) else []
        )

        if max(counts) if counts.size > 0 else 0 >= 3:
            three_val = np.argmax(counts) + 1
            combinations.append(f"Three of a Kind ({three_val}s)")

        if max(counts) if counts.size > 0 else 0 >= 4:
            four_val = np.argmax(counts) + 1
            combinations.append(f"Four of a Kind ({four_val}s)")

        if max(counts) if counts.size > 0 else 0 == 5:
            yahtzee_val = np.argmax(counts) + 1
            combinations.append(f"Yahtzee ({yahtzee_val}s)")

        # Check for full house
        if len(counts) >= 2 and sorted(counts, reverse=True)[:2] == [
            3,
            2,
        ]:
            three_val = np.where(counts == 3)[0][0] + 1
            two_val = np.where(counts == 2)[0][0] + 1
            combinations.append(
                f"Full House ({three_val}s over {two_val}s)"
            )

        sorted_unique = np.unique(dice_values)
        for straight in [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]:
            if all(x in sorted_unique for x in straight):
                combinations.append(
                    f"Small Straight ({'-'.join(map(str, straight))})"
                )
                break

        if len(sorted_unique) == 5 and (
            all(x in sorted_unique for x in [1, 2, 3, 4, 5])
            or all(x in sorted_unique for x in [2, 3, 4, 5, 6])
        ):
            straight = (
                [1, 2, 3, 4, 5]
                if sorted_unique[0] == 1
                else [2, 3, 4, 5, 6]
            )
            combinations.append(
                f"Large Straight ({'-'.join(map(str, straight))})"
            )

        if combinations:
            print("\nPossible Combinations:")
            for combo in combinations:
                print(f"• {combo}")

    # Get Q-values from agent and mask invalid actions
    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask

    # Sort by Q-value
    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Actions and Expected Values:")
    for i, (action_idx, value) in enumerate(valid_q[:num_top], 1):
        action = env.action_mapper.index_to_action(action_idx)
        if action.kind == ActionType.ROLL:
            print(f"{i}. Roll all dice (EV: {value:.1f})")
        elif action.kind == ActionType.HOLD:
            held = [
                i + 1 for i, hold in enumerate(action.data) if hold
            ]
            if held:
                held_values = [
                    state.current_dice[i - 1] for i in held
                ]
                print(
                    f"{i}. Hold {', '.join(f'{pos}({val})' for pos, val in zip(held, held_values))} (EV: {value:.1f})"
                )
            else:
                print(f"{i}. Roll all dice (EV: {value:.1f})")
        else:
            points = env.calc_score(action.data, state.current_dice)
            print(
                f"{i}. Score {action.data.name} for {points} points (EV: {value:.1f})"
            )

    return state, valid_q[:num_top], q_values


def evaluate_performance(
    agent: YahtzeeAgent, num_games: int = 50
) -> None:
    """
    Evaluate agent performance over multiple games and display statistics.
    """
    env = YahtzeeEnv(reward_strategy=RewardStrategy.STRATEGIC)
    encoder = StateEncoder(use_opponent_value=False)

    # Store original epsilon
    old_eps = agent.epsilon
    agent.epsilon = 0.01

    scores = []
    upper_bonuses = []
    yahtzees = []
    category_scores = {}

    # Set up progress bar
    progress_bar = tqdm(range(num_games), desc="Evaluating")

    # Run evaluation games
    for _ in progress_bar:
        state = env.reset()
        done = False

        # Reset category tracking for this game
        game_categories = {}
        got_yahtzee = False

        while not done:
            state_vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()
            action_idx = agent.select_action(state_vec, valid_actions)

            next_state, _, done, info = env.step(action_idx)
            state = next_state

            # Track categories scored
            if "category_scored" in info:
                category = info["category_scored"]
                points = info["points_scored"]
                game_categories[category] = points

                # Track Yahtzees
                if category == "YAHTZEE" and points == 50:
                    got_yahtzee = True

        # Calculate final score
        upper_total = 0
        for i, (cat, score) in enumerate(state.score_sheet.items()):
            if i < 6 and score is not None:
                upper_total += score

        bonus = 35 if upper_total >= 63 else 0
        upper_bonuses.append(1 if bonus > 0 else 0)

        total_score = (
            sum(
                score
                for score in state.score_sheet.values()
                if score is not None
            )
            + bonus
        )

        scores.append(total_score)
        yahtzees.append(1 if got_yahtzee else 0)

        # Update category stats
        for cat, score in game_categories.items():
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(score)

    # Restore epsilon
    agent.epsilon = old_eps

    # Calculate score brackets
    brackets = [
        (0, 100),
        (100, 150),
        (150, 200),
        (200, 250),
        (250, 300),
        (300, float("inf")),
    ]
    print("\nScore Distribution:")
    for low, high in brackets:
        count = np.sum(
            (np.array(scores) >= low) & (np.array(scores) < high)
        )
        percentage = (count / num_games) * 100
        high_str = f"{high:.0f}" if high != float("inf") else "inf"
        print(
            f"• {low:3.0f}-{high_str:>3}: {count:3.0f} games ({percentage:4.1f}%)"
        )

    # Plot score distribution
    plt.figure(figsize=(12, 6))

    mean_score = np.mean(scores)
    median_score = np.median(scores)

    plt.subplot(1, 2, 1)
    plt.hist(
        scores, bins=20, color="blue", alpha=0.7, edgecolor="black"
    )
    plt.axvline(
        mean_score,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_score:.1f}",
    )
    plt.axvline(
        median_score,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: {median_score:.1f}",
    )
    plt.xlabel("Score")
    plt.ylabel("Number of Games")
    plt.title("Score Distribution")
    plt.legend()

    # Show category performance
    plt.subplot(1, 2, 2)

    cats = []
    avg_scores = []
    for cat, cat_scores in category_scores.items():
        cats.append(cat)
        avg_scores.append(np.mean(cat_scores))

    # Sort categories by average score
    sorted_indices = np.argsort(avg_scores)
    cats = [cats[i] for i in sorted_indices]
    avg_scores = [avg_scores[i] for i in sorted_indices]

    plt.barh(cats, avg_scores, color="green", alpha=0.7)
    plt.xlabel("Average Score")
    plt.title("Category Performance")
    plt.tight_layout()

    # Save the plot if desired
    plt.savefig("agent_performance.png")
    plt.show()

    # Print key statistics
    print("\nPerformance Statistics:")
    print(f"• Games Played: {num_games}")
    print(f"• Mean Score: {mean_score:.1f}")
    print(f"• Median Score: {median_score:.1f}")
    print(f"• Min Score: {np.min(scores):.1f}")
    print(f"• Max Score: {np.max(scores):.1f}")
    print(f"• Standard Deviation: {np.std(scores):.1f}")
    print(f"• Upper Bonus Rate: {np.mean(upper_bonuses) * 100:.1f}%")
    print(f"• Yahtzee Rate: {np.mean(yahtzees) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Yahtzee RL Interactive Play"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games to play in performance evaluation mode",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["win", "score"],
        default="score",
        help="Training objective used for the model",
    )

    args = parser.parse_args()

    # Load agent
    agent = load_agent(args.model, args.objective)
    current_objective = args.objective

    # Main menu loop
    while True:
        print("\n=== Yahtzee RL Interactive Interface ===")
        print(f"Current model: {args.model or 'Selected model'}")
        print(f"Current objective: {current_objective}")

        print("\nSelect a mode:")
        print("• [1] Simulation Mode - Watch agent play a full game")
        print(
            "• [2] Calculation Mode - Analyze expected values for each action"
        )
        print(
            f"• [3] Performance Stats - View agent's statistics over {args.num_games} games"
        )
        print("• [4] Load Different Model")
        print("• [5] Change Objective")
        print("• [6] Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            print("\n=== Simulation Mode ===")
            print("Watch the agent play a full game of Yahtzee.")
            print(
                "The agent will show its decision-making process for each move."
            )
            input("Press Enter to start the game...")
            simulate_game(agent)
            input("\nPress Enter to return to menu...")

        elif choice == "2":
            print("\n=== Calculation Mode ===")
            print(
                "Analyze the expected value of each possible action in any game state."
            )
            current_state = None

            while True:
                current_state, valid_actions, q_values = (
                    show_action_values(agent, current_state)
                )

                print("\nOptions:")
                print("• [1] Take an action and continue analysis")
                print("• [2] Start a new game")
                print("• [3] Return to main menu")

                subchoice = input(
                    "\nEnter your choice (1-3): "
                ).strip()

                if subchoice == "1":
                    try:
                        action_num = int(
                            input(
                                "\nEnter action number to simulate: "
                            ).strip()
                        )
                        if 1 <= action_num <= len(valid_actions):
                            action_idx = valid_actions[
                                action_num - 1
                            ][0]
                            env = YahtzeeEnv()
                            current_state, reward, done, _ = env.step(
                                action_idx
                            )

                            if done:
                                print(
                                    f"\n=== Game Over! Final Score: {reward:.0f} ==="
                                )
                                break
                        else:
                            print(
                                "\nInvalid action number! Please try again."
                            )
                    except ValueError:
                        print("\nPlease enter a valid number!")

                elif subchoice == "2":
                    current_state = None

                elif subchoice == "3":
                    break

                else:
                    print("\nInvalid choice! Please try again.")

        elif choice == "3":
            print("\n=== Performance Statistics ===")
            print(
                f"Running {args.num_games} full games to evaluate agent performance..."
            )
            evaluate_performance(agent, args.num_games)
            input("\nPress Enter to return to menu...")

        elif choice == "4":
            print("\n=== Load New Model ===")
            agent = load_agent(
                objective=current_objective
            )  # Will prompt for model selection
            args.model = None  # Reset model path
            print("\nNew model loaded successfully!")

        elif choice == "5":
            print("\n=== Change Objective ===")
            print("Current objective: " + current_objective)
            print("Options:")
            print("• [1] Score optimization (maximize score)")
            print("• [2] Win rate optimization (beat opponent)")

            obj_choice = input("\nEnter your choice (1-2): ").strip()
            if obj_choice == "1":
                if current_objective != "score":
                    current_objective = "score"
                    agent = load_agent(args.model, current_objective)
                    print("Objective changed to score optimization")
                else:
                    print("Already using score optimization")
            elif obj_choice == "2":
                if current_objective != "win":
                    current_objective = "win"
                    agent = load_agent(args.model, current_objective)
                    print(
                        "Objective changed to win rate optimization"
                    )
                else:
                    print("Already using win rate optimization")
            else:
                print("\nInvalid choice! Objective unchanged.")

        elif choice == "6":
            print(
                "\nExiting Yahtzee RL Interactive Interface. Goodbye!"
            )
            break

        else:
            print(
                "\nInvalid choice! Please enter a number between 1 and 6."
            )


if __name__ == "__main__":
    main()
