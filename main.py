import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.cuda.amp as amp
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
    base_reward = score  # Keep original score as base
    bonus_reward = 0.0

    # Get current dice state for analysis
    dice = env.state.current_dice
    counts = np.bincount(dice)[1:] if any(dice) else []
    max_count = max(counts) if any(counts) else 0
    unique_vals = np.unique(dice[dice > 0]) if any(dice) else []

    # Track upper section progress
    upper_cats = [
        YahtzeeCategory.ONES,
        YahtzeeCategory.TWOS,
        YahtzeeCategory.THREES,
        YahtzeeCategory.FOURS,
        YahtzeeCategory.FIVES,
        YahtzeeCategory.SIXES,
    ]
    upper_score = sum(env.state.score_sheet[cat] or 0 for cat in upper_cats)
    upper_remaining = sum(1 for cat in upper_cats if env.state.score_sheet[cat] is None)

    # Early game strategy (first 6 turns)
    num_scored = sum(1 for score in env.state.score_sheet.values() if score is not None)
    is_early_game = num_scored < 6

    # Add small baseline reward
    bonus_reward += 2.0  # Small baseline to avoid too many negatives

    # Yahtzee opportunity rewards
    if max_count >= 4:
        if env.state.score_sheet[YahtzeeCategory.YAHTZEE] is None:
            bonus_reward += 8.0  # Major bonus for potential Yahtzee when category open
        else:
            bonus_reward += 4.0  # Still good for other categories
    elif max_count == 3:
        bonus_reward += 3.0  # Good potential for Yahtzee or other high scores

    # Upper section strategy
    if upper_remaining > 0:
        points_needed = max(0, 63 - upper_score)  # Points needed for bonus
        avg_needed = points_needed / upper_remaining if upper_remaining > 0 else 0

        if category in upper_cats:
            val = upper_cats.index(category) + 1  # Value for this category (1-6)
            if score >= val * 3:  # Got 3 or more of the number
                bonus_reward += 5.0
            elif score >= avg_needed:
                bonus_reward += 3.0  # Good progress toward bonus
            elif score > 0:
                bonus_reward += 1.0  # Any progress is good

            # Extra reward for higher numbers in upper section
            bonus_reward += val * 0.2  # Small scaling bonus for higher numbers

    # Straight opportunity rewards
    if len(unique_vals) >= 4:
        # Check for potential straights
        sorted_vals = np.sort(unique_vals)
        gaps = np.diff(sorted_vals)
        if np.all(gaps == 1):  # Sequential values
            if env.state.score_sheet[YahtzeeCategory.LARGE_STRAIGHT] is None:
                bonus_reward += 4.0
            elif env.state.score_sheet[YahtzeeCategory.SMALL_STRAIGHT] is None:
                bonus_reward += 3.0

    # Penalties for suboptimal plays
    if score == 0:  # Scoring zero
        if category == YahtzeeCategory.CHANCE:
            bonus_reward -= 10.0  # Never zero Chance - it's a safety net
        elif max_count >= 3:
            bonus_reward -= 8.0  # Wasting three of a kind
        elif len(unique_vals) >= 4:
            bonus_reward -= 6.0  # Wasting straight opportunity
        elif is_early_game and category in upper_cats:
            bonus_reward -= 4.0  # Zeroing upper section early is usually bad
    else:
        # Small reward for any non-zero score
        bonus_reward += 1.0

    # Achievement bonuses
    if category == YahtzeeCategory.YAHTZEE and score == 50:
        bonus_reward += 10.0  # Yahtzee is highest priority
    elif category == YahtzeeCategory.LARGE_STRAIGHT and score == 40:
        bonus_reward += 6.0
    elif category == YahtzeeCategory.SMALL_STRAIGHT and score == 30:
        bonus_reward += 5.0
    elif category == YahtzeeCategory.FULL_HOUSE and score == 25:
        bonus_reward += 4.0
    elif category == YahtzeeCategory.FOUR_OF_A_KIND and score >= 20:
        bonus_reward += 3.0
    elif category == YahtzeeCategory.THREE_OF_A_KIND and score >= 20:
        bonus_reward += 2.0

    # Late game adjustments
    if num_scored >= 10:  # Last few turns
        if category == YahtzeeCategory.CHANCE and score > 20:
            bonus_reward += 2.0  # Reward good Chance scores late
        elif score > 0:
            bonus_reward += 1.0  # Small bonus for any non-zero late game

    # Keep rewards in reasonable range
    final_reward = base_reward + bonus_reward
    return max(final_reward, -10.0)  # Limit negative rewards


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
        "scheduler": agent.scheduler.state_dict(),
        "metrics": metrics,
        "epsilon": agent.epsilon,
    }

    if is_best:
        # Use mean of last 100 episodes if eval_score not available
        if "eval_score" in metrics:
            score = int(metrics["eval_score"])
        else:
            score = int(np.mean(metrics["episode_rewards"][-100:]))
        filename = f"{models_dir}/yahtzee_run_{run_id}_score{score}.pth"
    else:
        filename = f"{models_dir}/yahtzee_run_{run_id}_checkpoint_{episode}.pth"

    torch.save(checkpoint, filename)
    print(f"Saved model to: {filename}")
    return filename


def train(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 20000,
    checkpoint_freq: int = 100,
    eval_freq: int = 50,
    num_eval_episodes: int = 50,
    patience: int = 3,
    min_improvement: float = 0.5,
    num_workers: int = 8,
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
        "num_workers": num_workers,
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

    # Create agent with optimized hyperparameters for faster training
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=NUM_ACTIONS,
        batch_size=2048,  # Large batch size for GPU utilization
        gamma=0.99,
        learning_rate=3e-4,  # Higher learning rate for faster convergence
        target_update=50,
        device=device,
    )

    # Enable mixed precision training
    scaler = amp.GradScaler()

    start_episode = 0
    metrics = {
        "best_eval_score": float("-inf"),
        "patience_counter": 0,
        "episode_rewards": [],
        "eval_scores": [],
        "losses": [],
        "current_eval_score": None,  # Add this to track current eval score
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

    # Create worker processes for parallel environment stepping
    worker_envs = [YahtzeeEnv() for _ in range(num_workers)]
    worker_encoders = [StateEncoder() for _ in range(num_workers)]

    last_save_time = time.time()
    progress = tqdm(range(start_episode, num_episodes), desc="Training")

    try:
        for episode in progress:
            # Run parallel episodes with automatic mixed precision
            episode_rewards = []
            episode_losses = []

            for worker_id in range(num_workers):
                state = worker_envs[worker_id].reset()
                total_reward = 0
                worker_losses = []

                while True:
                    state_vec = worker_encoders[worker_id].encode(state)
                    valid_actions = worker_envs[worker_id].get_valid_actions()

                    if not valid_actions:
                        break

                    with amp.autocast():
                        action_idx = agent.select_action(state_vec, valid_actions)
                        action = worker_envs[worker_id].idx_to_action[action_idx]
                        next_state, reward, done, _ = worker_envs[worker_id].step(
                            action_idx
                        )

                        if action.kind == ActionType.SCORE:
                            reward = calculate_strategic_reward(
                                worker_envs[worker_id], action.data, reward
                            )

                        next_state_vec = worker_encoders[worker_id].encode(next_state)

                        # Scale loss for mixed precision training
                        with amp.autocast():
                            loss = agent.train_step(
                                state_vec,
                                action_idx,
                                reward,
                                next_state_vec,
                                done,
                            )

                    if loss is not None:
                        worker_losses.append(loss)

                    state = next_state
                    total_reward += reward

                    if done:
                        break

                episode_rewards.append(total_reward)
                if worker_losses:
                    episode_losses.append(np.mean(worker_losses))

            # Update metrics and logging
            metrics["episode_rewards"].extend(episode_rewards)
            if episode_losses:
                metrics["losses"].extend(episode_losses)

            mean_reward = np.mean(episode_rewards)
            mean_loss = np.mean(episode_losses) if episode_losses else None

            wandb.log(
                {
                    "episode": episode,
                    "reward": mean_reward,
                    "epsilon": agent.epsilon,
                    "loss": mean_loss,
                    "learning_rate": agent.optimizer.param_groups[0]["lr"],
                },
                step=episode,
            )

            # Show progress
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

            # Quick evaluation
            if (episode + 1) % eval_freq == 0:
                agent.eval()
                eval_rewards = []

                # Parallel evaluation
                for _ in range(num_eval_episodes // num_workers):
                    worker_rewards = []
                    for worker_id in range(num_workers):
                        eval_reward = evaluate_episode(
                            agent, worker_envs[worker_id], worker_encoders[worker_id]
                        )
                        worker_rewards.append(eval_reward)
                    eval_rewards.extend(worker_rewards)

                mean_eval_score = np.mean(eval_rewards)
                metrics["eval_scores"].append(mean_eval_score)
                metrics["current_eval_score"] = (
                    mean_eval_score  # Update current eval score
                )

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
                        print(
                            f"\nEarly stopping after {patience} evaluations without improvement"
                        )
                        break

                agent.train()

            # Save checkpoint every 30 minutes
            current_time = time.time()
            if current_time - last_save_time >= 1800:
                filename = save_checkpoint(agent, episode + 1, run_id, metrics)
                wandb.save(filename)
                last_save_time = current_time

        # Always save final model, regardless of how we exit the loop
        print("\nSaving final model...")
        final_metrics = {
            **metrics,
            "final_mean_reward": np.mean(metrics["episode_rewards"][-100:]),
        }
        filename = save_checkpoint(agent, episode + 1, run_id, final_metrics)
        wandb.save(filename)
        print(f"Final model saved to: {filename}")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Save final state
        try:
            print("\nSaving final state...")
            final_metrics = {
                **metrics,
                "final_mean_reward": np.mean(metrics["episode_rewards"][-100:]),
                "interrupted": True,
            }
            filename = save_checkpoint(agent, episode + 1, run_id, final_metrics)
            wandb.save(filename)
            print(f"Final state saved to: {filename}")
        except Exception as e:
            print(f"Error saving final state: {str(e)}")

        # Cleanup wandb
        if wandb.run is not None:
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
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers"
    )
    args = parser.parse_args()

    # If run_id provided, try to find latest checkpoint
    checkpoint_path = args.checkpoint
    if args.run_id and not checkpoint_path:
        checkpoint_path = get_latest_checkpoint(args.run_id)
        if checkpoint_path:
            print(f"Found checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoints found for run {args.run_id}")

    try:
        train(
            run_id=args.run_id,
            checkpoint_path=checkpoint_path,
            num_episodes=args.episodes,
            num_workers=args.workers,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Cleanup wandb
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
