import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import wandb
from IPython.display import clear_output
from tqdm import tqdm

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import (
    IDX_TO_ACTION,
    NUM_ACTIONS,
    ActionType,
    GameState,
    YahtzeeCategory,
    YahtzeeEnv,
)
from utils import plot_training_progress


def evaluate_agent(agent: YahtzeeAgent, num_games: int = 100) -> dict:
    """Evaluate agent performance across multiple games."""
    env = YahtzeeEnv()
    encoder = StateEncoder()
    scores = []

    # Store original state and set to eval mode
    was_training = agent.training_mode
    agent.eval()

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

            action_idx = agent.select_action(state_vec, valid_actions)
            state, reward, done, _ = env.step(action_idx)
            total_reward += reward

        scores.append(total_reward)

    # Restore original state
    if was_training:
        agent.train()

    # Calculate statistics
    scores = np.array(scores)
    stats = {
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "max": np.max(scores),
        "min": np.min(scores),
    }

    print(
        f"\nEval Mean: {stats['mean']:.1f} ± {stats['std']:.1f}, Max: {stats['max']:.1f}, Min: {stats['min']:.1f}"
    )
    return stats


def quick_validation_training(
    num_episodes: int = 5000,  # Reduced episodes for quick validation
    batch_size: int = 1024,  # Larger batch size to improve GPU utilization
    eval_games: int = 20,  # Fewer eval games
    eval_interval: int = 500,  # More frequent evaluation
) -> tuple:
    """Quick training loop for validating the approach."""
    # Generate unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yahtzee_quick_val_{timestamp}"

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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Training metrics
    all_rewards = []
    best_mean = 0
    eval_stats = []
    best_model_path = None

    # Create models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

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
            progress.set_postfix({"Avg": f"{avg_score:.1f}"})

        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            print(f"\nQuick evaluation at episode {episode + 1}...")
            stats = evaluate_agent(agent, num_games=eval_games)
            eval_stats.append(stats)

            if stats["mean"] > best_mean:
                best_mean = stats["mean"]
                # Save only the best model with score in filename
                model_path = os.path.join(
                    models_dir, f"{run_name}_score{int(stats['mean'])}.pth"
                )
                agent.save(model_path)
                # Update best model path
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)  # Remove old best model
                best_model_path = model_path

            plot_training_progress(
                all_rewards,
                window=50,
                title=(
                    f"Validation Progress (Episode {episode + 1})\n"
                    f"Best Mean: {best_mean:.1f}"
                ),
            )

    return agent, all_rewards, eval_stats


def get_latest_checkpoint(run_id: str) -> Optional[str]:
    """Find the latest checkpoint for a given run ID."""
    models_dir = "models"
    checkpoints = [
        f
        for f in os.listdir(models_dir)
        if f.startswith(f"yahtzee_run_{run_id}_checkpoint_")
    ]
    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(models_dir, checkpoints[-1])


def calculate_strategic_reward(
    env: YahtzeeEnv, category: YahtzeeCategory, score: float
) -> float:
    """Calculate strategic reward based on Yahtzee best practices."""
    base_reward = score
    bonus_reward = 0.0

    # Track upper section progress
    upper_score = sum(
        env.state.score_sheet[cat] or 0
        for cat in [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]
    )

    # Count remaining upper categories
    upper_remaining = sum(
        1
        for cat in [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]
        if env.state.score_sheet[cat] is None
    )

    if upper_remaining > 0:
        points_needed = max(0, 63 - upper_score)
        avg_needed = points_needed / upper_remaining

        # Reward for good upper section progress
        if category in [
            YahtzeeCategory.ONES,
            YahtzeeCategory.TWOS,
            YahtzeeCategory.THREES,
            YahtzeeCategory.FOURS,
            YahtzeeCategory.FIVES,
            YahtzeeCategory.SIXES,
        ]:
            if score >= avg_needed:
                bonus_reward += 5.0  # Bonus for meeting/exceeding average needed
            elif score > 0:
                bonus_reward += 2.0  # Small bonus for any positive score

    # Extra rewards for key achievements
    if category == YahtzeeCategory.YAHTZEE and score == 50:
        bonus_reward += 10.0  # Big bonus for Yahtzee
    elif category == YahtzeeCategory.LARGE_STRAIGHT and score == 40:
        bonus_reward += 5.0  # Bonus for Large Straight
    elif category == YahtzeeCategory.SMALL_STRAIGHT and score == 30:
        bonus_reward += 3.0  # Bonus for Small Straight
    elif category == YahtzeeCategory.FULL_HOUSE and score == 25:
        bonus_reward += 2.0  # Bonus for Full House

    # Penalty for wasting good combinations
    if score == 0 and category not in [YahtzeeCategory.CHANCE]:
        counts = (
            np.bincount(env.state.current_dice)[1:]
            if any(env.state.current_dice)
            else []
        )
        if any(c >= 3 for c in counts) or len(np.unique(env.state.current_dice)) >= 4:
            bonus_reward -= 5.0  # Penalty for wasting good combinations

    return base_reward + bonus_reward


def load_checkpoint(
    checkpoint_path: str, agent: YahtzeeAgent, device: torch.device
) -> int:
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    try:
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint["episode"]
    except KeyError:
        print("Loading legacy checkpoint format...")
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(checkpoint)
        try:
            start_episode = int(checkpoint_path.split("_")[-1].split(".")[0])
        except (ValueError, IndexError):
            start_episode = 0

    print(f"Resuming from episode {start_episode}")
    return start_episode


def save_checkpoint(
    agent: YahtzeeAgent,
    episode: int,
    run_id: str,
    metrics: dict,
    is_best: bool = False,
) -> str:
    """Save checkpoint with consistent format and metrics."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Save full training state
    checkpoint = {
        "episode": episode,
        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "metrics": metrics,
        "epsilon": agent.epsilon,
    }

    if is_best:
        score = int(metrics["eval_score"])
        filename = f"{models_dir}/yahtzee_run_{run_id}_score{score}.pth"
    else:
        filename = f"{models_dir}/yahtzee_run_{run_id}_checkpoint_{episode}.pth"

    torch.save(checkpoint, filename)
    print(f"Saved model to: {filename}")
    return filename


def train(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 100000,
    checkpoint_freq: int = 5000,
    eval_freq: int = 1000,
    num_eval_episodes: int = 100,
    patience: int = 5,
    min_improvement: float = 1.0,
) -> None:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        "num_episodes": num_episodes,
        "checkpoint_freq": checkpoint_freq,
        "eval_freq": eval_freq,
        "num_eval_episodes": num_eval_episodes,
        "patience": patience,
        "min_improvement": min_improvement,
    }

    wandb.init(
        project="yahtzee-rl",
        name=f"yahtzee_run_{run_id}",
        config=config,
        resume=checkpoint_path is not None,
    )

    env = YahtzeeEnv()
    encoder = StateEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adjusted hyperparameters for improved performance
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=NUM_ACTIONS,
        batch_size=1024,  # Increased batch size
        gamma=0.995,  # Slightly higher discount factor
        learning_rate=1e-4,  # Lower learning rate for stability
        target_update=250,  # More frequent target updates
        device=device,
    )

    start_episode = 0
    metrics = {
        "best_eval_score": float("-inf"),
        "patience_counter": 0,
        "episode_rewards": [],
        "eval_scores": [],
        "losses": [],
    }

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        metrics.update(checkpoint.get("metrics", {}))
        start_episode = checkpoint["episode"]
        print(f"Resuming from episode {start_episode}")

    last_save_time = time.time()
    progress = tqdm(range(start_episode, num_episodes), desc="Training")

    for episode in progress:
        state = env.reset()
        total_reward = 0
        episode_losses = []

        while True:
            state_vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()

            if not valid_actions:
                break

            action_idx = agent.select_action(state_vec, valid_actions)
            action = env.idx_to_action[action_idx]
            next_state, reward, done, _ = env.step(action_idx)

            if action.kind == ActionType.SCORE:
                reward = calculate_strategic_reward(env, action.data, reward)

            next_state_vec = encoder.encode(next_state)
            loss = agent.train_step(
                state_vec,
                action_idx,
                reward,
                next_state_vec,
                done,
            )

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        metrics["episode_rewards"].append(total_reward)
        if episode_losses:
            metrics["losses"].append(np.mean(episode_losses))

        wandb.log(
            {
                "episode": episode,
                "reward": total_reward,
                "epsilon": agent.epsilon,
                "loss": np.mean(episode_losses) if episode_losses else None,
            },
            step=episode,
        )

        if len(metrics["episode_rewards"]) >= 100:
            mean_100 = np.mean(metrics["episode_rewards"][-100:])
            best_score = metrics["best_eval_score"]
            patience = metrics["patience_counter"]
            progress.set_postfix(
                {
                    "Mean100": f"{mean_100:.1f}",
                    "Best": f"{best_score:.1f}",
                    "Patience": patience,
                }
            )

        if (episode + 1) % eval_freq == 0:
            agent.eval()
            eval_rewards = []

            for _ in range(num_eval_episodes):
                eval_reward = evaluate_episode(agent, env, encoder)
                eval_rewards.append(eval_reward)

            mean_eval_score = np.mean(eval_rewards)
            metrics["eval_scores"].append(mean_eval_score)

            wandb.log(
                {
                    "eval_score": mean_eval_score,
                    "eval_score_std": np.std(eval_rewards),
                },
                step=episode,
            )

            if mean_eval_score > metrics["best_eval_score"] + min_improvement:
                metrics["best_eval_score"] = mean_eval_score
                metrics["patience_counter"] = 0
                filename = save_checkpoint(agent, episode, run_id, metrics, True)
                wandb.save(filename)
                print(f"\nNew best score: {metrics['best_eval_score']:.1f}")
            else:
                metrics["patience_counter"] += 1
                if metrics["patience_counter"] >= patience:
                    msg = (
                        f"\nEarly stopping after {patience} evaluations "
                        "without improvement"
                    )
                    print(msg)
                    break

            agent.train()

        current_time = time.time()
        if current_time - last_save_time >= 1800:
            filename = save_checkpoint(agent, episode + 1, run_id, metrics)
            wandb.save(filename)
            last_save_time = current_time

        if (episode + 1) % (eval_freq * 2) == 0:
            title = (
                f"Training Progress (Episode {episode+1})\n"
                f"Best: {metrics['best_eval_score']:.1f}"
            )
            plot_training_progress(
                metrics["episode_rewards"],
                window=100,
                title=title,
            )

    wandb.finish()


def evaluate_episode(
    agent: YahtzeeAgent, env: YahtzeeEnv, encoder: StateEncoder
) -> float:
    """Run a single evaluation episode."""
    state = env.reset()
    total_reward = 0
    done = False

    # Store original epsilon and set to minimum for deterministic evaluation
    old_eps = agent.epsilon
    agent.epsilon = 0.02

    while not done:
        state_vec = encoder.encode(state)
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            break

        action_idx = agent.select_action(state_vec, valid_actions)
        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

    # Restore original epsilon
    agent.epsilon = old_eps
    return total_reward


def simulate_game(agent: YahtzeeAgent, render: bool = True) -> float:
    """Simulate a single game with visualization."""
    env = YahtzeeEnv()
    encoder = StateEncoder()
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    # Store original epsilon
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

        action_idx = agent.select_action(state_vec, valid_actions)
        action = IDX_TO_ACTION[action_idx]

        if render:
            print("\nAgent's decision:")
            if action.kind == ActionType.ROLL:
                print("Action: ROLL all dice")
            elif action.kind == ActionType.HOLD_MASK:
                held = [i + 1 for i, hold in enumerate(action.data) if hold]
                print(f"Action: Hold dice at positions {held}")
            else:
                print(f"Action: Score {action.data.name}")

        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        if render:
            if action.kind == ActionType.SCORE:
                print(f"Scored {reward:.1f} points")
                turn += 1
            time.sleep(1)

    if render:
        clear_output(wait=True)
        print("\n=== Game Over ===")
        print(env.render())
        print(f"\nFinal Score: {total_reward:.1f}")

    # Restore epsilon
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
    """Main training function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Yahtzee agent")
    parser.add_argument("--run_id", type=str, help="Run ID for resuming training")
    parser.add_argument(
        "--episodes", type=int, default=100000, help="Number of episodes"
    )
    args = parser.parse_args()

    # If run_id provided, try to find latest checkpoint
    checkpoint_path = None
    if args.run_id:
        checkpoint_path = get_latest_checkpoint(args.run_id)
        if checkpoint_path:
            print(f"Found checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoints found for run {args.run_id}")

    train(
        run_id=args.run_id,
        checkpoint_path=checkpoint_path,
        num_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
