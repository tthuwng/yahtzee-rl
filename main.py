import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from IPython.display import clear_output
from tqdm import tqdm

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import NUM_ACTIONS, ActionType, GameState, YahtzeeEnv
from utils import is_colab, plot_training_progress, save_to_colab


def evaluate_agent(agent: YahtzeeAgent, num_games: int = 100) -> dict:
    """
    Evaluate agent performance across multiple games.
    Returns dict with detailed statistics.
    """
    env = YahtzeeEnv()
    encoder = StateEncoder()
    scores = []

    # Store original temperature/epsilon
    if agent.use_boltzmann:
        old_temp = agent.temperature
        agent.temperature = 0.01  # Nearly deterministic
    else:
        old_eps = agent.epsilon
        agent.epsilon = 0.01

    # Run evaluation games
    for _ in range(num_games):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            # Use greedy action selection
            action_idx = agent.select_action_greedy(state_vec, valid_actions)
            state, reward, done, _ = env.step(action_idx)
            total_reward += reward

        scores.append(total_reward)

    # Restore exploration parameters
    if agent.use_boltzmann:
        agent.temperature = old_temp
    else:
        agent.epsilon = old_eps

    # Calculate statistics
    scores = np.array(scores)
    stats = {
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "scores": scores,
    }

    # Print detailed results
    print("\nEvaluation Results:")
    print(f"Mean Score: {stats['mean']:.1f}")
    print(f"Median Score: {stats['median']:.1f}")
    print(f"Std Dev: {stats['std']:.1f}")
    print(f"Min Score: {stats['min']:.1f}")
    print(f"Max Score: {stats['max']:.1f}")

    # Plot score distribution
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=20, edgecolor="black")
    plt.axvline(
        stats["mean"],
        color="red",
        linestyle="dashed",
        label=f"Mean ({stats['mean']:.1f})",
    )
    plt.axvline(
        stats["median"],
        color="green",
        linestyle="dashed",
        label=f"Median ({stats['median']:.1f})",
    )
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    return stats


def quick_validation_training(
    num_episodes: int = 5000,  # Reduced episodes for quick validation
    batch_size: int = 1024,  # Larger batch size to improve GPU utilization
    eval_games: int = 20,  # Fewer eval games
    eval_interval: int = 500,  # More frequent evaluation
) -> tuple:
    """Quick training loop for validating the approach."""
    env = YahtzeeEnv()
    encoder = StateEncoder()

    # Initialize agent with modified parameters
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=NUM_ACTIONS,
        batch_size=batch_size,
        gamma=0.99,
        learning_rate=2e-4,  # Slightly higher learning rate
        target_update=250,  # More frequent target updates
        use_boltzmann=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Training metrics
    all_rewards = []
    best_mean = 0
    eval_stats = []

    # Training loop with progress bar
    progress = tqdm(range(num_episodes), desc="Validation Training")
    for episode in progress:
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action_idx = agent.select_action(state_vec, valid_actions)
            next_state, reward, done, _ = env.step(action_idx)
            next_state_vec = encoder.encode(next_state)

            # Train DQN
            agent.train_step(
                state_vec,
                action_idx,
                reward,
                next_state_vec,
                done,
            )

            total_reward += reward
            state = next_state

        all_rewards.append(total_reward)

        # Update progress more frequently
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(all_rewards[-50:])
            progress.set_postfix(
                {"Avg": f"{avg_score:.1f}", "Temp": f"{agent.temperature:.3f}"}
            )

        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            print(f"\nQuick evaluation at episode {episode + 1}...")
            stats = evaluate_agent(agent, num_games=eval_games)
            eval_stats.append(stats)

            if stats["mean"] > best_mean:
                best_mean = stats["mean"]
                agent.save("quick_val_best.pth")

            plot_training_progress(
                all_rewards,
                window=50,
                title=(
                    f"Validation Progress (Episode {episode + 1})\n"
                    f"Best Mean: {best_mean:.1f}"
                ),
            )

    return agent, all_rewards, eval_stats


def train_yahtzee_agent(
    num_episodes: int = 50000,  # Reduced episodes
    use_boltzmann: bool = True,
    plot_interval: int = 1000,  # More frequent plotting
    eval_interval: int = 5000,  # More frequent evaluation
) -> tuple:
    """Train a Yahtzee agent and return the trained agent + reward history."""
    wandb.init(
        project="yahtzee-rl",
        config={
            "num_episodes": num_episodes,
            "use_boltzmann": use_boltzmann,
            "batch_size": 512,
            "gamma": 0.99,
            "learning_rate": 1e-4,
            "target_update": 500,
        },
    )

    env = YahtzeeEnv()
    encoder = StateEncoder()

    # Instantiate agent with optimized parameters
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=NUM_ACTIONS,
        batch_size=512,  # Smaller batch size
        gamma=0.99,  # Slightly lower discount
        learning_rate=1e-4,  # Higher learning rate
        target_update=500,  # More frequent updates
        use_boltzmann=use_boltzmann,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Training metrics
    all_rewards = []
    best_score = 0
    recent_scores = []
    eval_stats = []
    plateau_counter = 0
    best_mean = 0

    # Training loop with progress bar
    progress = tqdm(range(num_episodes), desc="Training")
    for episode in progress:
        state = env.reset()
        total_reward = 0.0
        done = False
        episode_loss = 0.0
        num_steps = 0

        while not done:
            state_vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action_idx = agent.select_action(state_vec, valid_actions)
            next_state, reward, done, _ = env.step(action_idx)
            next_state_vec = encoder.encode(next_state)

            loss = agent.train_step(
                state_vec,
                action_idx,
                reward,
                next_state_vec,
                done,
            )
            episode_loss += loss
            num_steps += 1

            total_reward += reward
            state = next_state

        avg_loss = episode_loss / max(num_steps, 1)
        all_rewards.append(total_reward)
        recent_scores.append(total_reward)
        if len(recent_scores) > 100:  # Shorter window
            recent_scores.pop(0)

        if total_reward > best_score:
            best_score = total_reward

        metrics = {
            "episode": episode,
            "reward": total_reward,
            "avg_loss": avg_loss,
            "best_score": best_score,
            "avg_score_100": np.mean(recent_scores),
            "temperature" if use_boltzmann else "epsilon": (
                agent.temperature if use_boltzmann else agent.epsilon
            ),
        }
        wandb.log(metrics)

        # Update progress bar more frequently
        if (episode + 1) % 50 == 0:  # More frequent updates
            avg_score = np.mean(recent_scores)
            if use_boltzmann:
                temp = agent.temperature
                progress.set_postfix(
                    {
                        "Avg": f"{avg_score:.1f}",
                        "Best": f"{best_score:.1f}",
                        "Temp": f"{temp:.3f}",
                    }
                )
            else:
                eps = agent.epsilon
                progress.set_postfix(
                    {
                        "Avg": f"{avg_score:.1f}",
                        "Best": f"{best_score:.1f}",
                        "Eps": f"{eps:.3f}",
                    }
                )

        # More frequent evaluation
        if (episode + 1) % eval_interval == 0:
            print(f"\nEvaluating at episode {episode + 1}...")
            stats = evaluate_agent(agent, num_games=50)  # Fewer eval games
            eval_stats.append(stats)

            eval_metrics = {
                "eval/mean_score": stats["mean"],
                "eval/median_score": stats["median"],
                "eval/std_score": stats["std"],
                "eval/min_score": stats["min"],
                "eval/max_score": stats["max"],
            }
            wandb.log(eval_metrics)

            if stats["mean"] > best_mean:
                best_mean = stats["mean"]
                plateau_counter = 0
                # Save best model so far
                agent.save("best_model.pth")
                wandb.save("best_model.pth")
            else:
                plateau_counter += 1

            if plateau_counter >= 3:
                print("\nTraining has plateaued. Consider stopping.")

            plot_training_progress(
                all_rewards,
                window=100,  # Shorter window
                title=f"Training Progress (Episode {episode + 1})\n"
                f"Best Eval Mean: {best_mean:.1f}",
            )

    wandb.finish()
    return agent, all_rewards, eval_stats


def simulate_game(agent: YahtzeeAgent, render: bool = True) -> float:
    """
    Simulate a single game with visualization of dice and scoresheet.
    Returns the final score.
    """
    env = YahtzeeEnv()
    encoder = StateEncoder()
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    # Store original temperature/epsilon
    if agent.use_boltzmann:
        old_temp = agent.temperature
        agent.temperature = 0.01  # Nearly deterministic
    else:
        old_eps = agent.epsilon
        agent.epsilon = 0.01

    while not done:
        if render:
            clear_output(wait=True)
            print(f"\n=== Turn {turn} ===")
            print(env.render())

        state_vec = encoder.encode(state)
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        # Get Q-values for all actions
        q_values = agent.get_q_values(state_vec)
        # Mask invalid actions
        mask = np.full(agent.action_size, float("-inf"))
        mask[valid_actions] = 0
        q_values = q_values + mask

        # Select best action
        action_idx = q_values.argmax()
        action = env.IDX_TO_ACTION[action_idx]

        if render:
            print("\nAgent's decision:")
            if action.kind == ActionType.ROLL:
                print("Action: ROLL all dice")
            elif action.kind == ActionType.HOLD_MASK:
                held = [i + 1 for i, hold in enumerate(action.data) if hold]
                print(f"Action: Hold dice at positions {held}")
            else:
                print(f"Action: Score {action.data.name}")
                print(f"Expected value: {q_values[action_idx]:.1f}")

        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        if render:
            if action.kind == ActionType.SCORE:
                print(f"Scored {reward:.1f} points")
                turn += 1
            time.sleep(1)  # Pause to make it easier to follow

    if render:
        clear_output(wait=True)
        print("\n=== Game Over ===")
        print(env.render())
        print(f"\nFinal Score: {total_reward:.1f}")

    # Restore exploration parameters
    if agent.use_boltzmann:
        agent.temperature = old_temp
    else:
        agent.epsilon = old_eps

    return total_reward


def show_action_values(
    agent: YahtzeeAgent, state: Optional[GameState] = None, num_top: int = 5
) -> tuple:
    """
    Show expected values for all valid actions in the current state.
    If state is None, starts a new game.
    Returns (state, valid_actions, q_values) for further use.
    """
    env = YahtzeeEnv()
    encoder = StateEncoder()

    if state is None:
        state = env.reset()

    print("\nCurrent Game State:")
    print(env.render())

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train with more episodes and evaluation:
    agent, rewards, eval_stats = train_yahtzee_agent(
        num_episodes=50000,
        use_boltzmann=True,
        plot_interval=1000,
        eval_interval=5000,
    )

    # Final evaluation and save best model
    print("\nFinal Evaluation:")
    stats = evaluate_agent(agent, num_games=100)
    print(f"\nFinal mean score: {stats['mean']:.1f}")
    print(f"Final median score: {stats['median']:.1f}")

    # Save model
    if is_colab():
        model_path = "/content/drive/MyDrive/yahtzee_dqn_improved.pth"
    else:
        model_path = "yahtzee_dqn_improved.pth"
    agent.save(model_path)
    print(f"Saved agent to {model_path}")
    save_to_colab(model_path)

    # Interactive mode
    env = YahtzeeEnv()  # For calculation mode
    current_state = None

    while True:
        print("\nChoose mode:")
        print("1. Simulation Mode (watch agent play)")
        print("2. Calculation Mode (see action values)")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            print("\nSimulating a game...")
            simulate_game(agent, render=True)
        elif choice == "2":
            print("\nStarting calculation mode...")
            current_state, valid_actions = show_action_values(agent)

            while True:
                print("\nOptions:")
                print("1. Take an action and continue")
                print("2. Start new game")
                print("3. Return to main menu")

                subchoice = input("\nEnter choice (1-3): ").strip()

                if subchoice == "1":
                    try:
                        prompt = "\nEnter action number: "
                        action_num = int(input(prompt).strip())
                        if 1 <= action_num <= len(valid_actions):
                            action_idx = valid_actions[action_num - 1][0]
                            next_state, reward, done, _ = env.step(action_idx)

                            if done:
                                msg = f"\nGame Over! Final Score: {reward:.1f}"
                                print(msg)
                                break
                            else:
                                # Get next state values
                                result = show_action_values(agent, current_state)
                                current_state, valid_actions = result
                        else:
                            print("\nInvalid action number!")
                    except ValueError:
                        print("\nPlease enter a valid number!")
                elif subchoice == "2":
                    current_state, valid_actions = show_action_values(agent)
                else:
                    break
        else:
            break

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
