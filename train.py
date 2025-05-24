#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import GameState, RewardStrategy, YahtzeeEnv
from yahtzee_types import ActionType


def evaluate_agent(
    agent: YahtzeeAgent, num_games: int = 100, objective: str = "win"
) -> dict:
    """
    Evaluate the agent by playing multiple games
    return performance metrics.
    """
    scores = []
    env = YahtzeeEnv(
        use_opponent_value=(objective == "win"),
        reward_strategy=RewardStrategy.STANDARD,
    )
    encoder = StateEncoder(use_opponent_value=(objective == "win"))

    # Store original epsilon and set to minimum for deterministic evaluation
    old_eps = agent.epsilon
    agent.epsilon = 0.02
    if (
        hasattr(agent.policy_net, "use_noisy")
        and agent.policy_net.use_noisy
    ):
        # Sample fresh noise for each evaluation episode
        agent.policy_net.sample_noise()

    # Run games
    for _ in range(num_games):
        state = env.reset()
        done = False
        episode_score = 0

        while not done:
            state_vec = encoder.encode(
                state,
                opponent_value=0.5 if objective == "win" else 0.0,
            )
            valid_actions = env.get_valid_actions()
            action_idx = agent.select_action(state_vec, valid_actions)
            state, reward, done, _ = env.step(action_idx)
            episode_score += reward

        # Calculate final score the standard way for consistent reporting
        upper_cats = [
            cat
            for i, cat in enumerate(env.state.score_sheet.keys())
            if i < 6
        ]
        upper_score = sum(
            env.state.score_sheet[cat] or 0 for cat in upper_cats
        )
        upper_bonus = 35 if upper_score >= 63 else 0
        total_score = (
            sum(
                score
                for score in env.state.score_sheet.values()
                if score is not None
            )
            + upper_bonus
        )
        scores.append(total_score)

    # Calculate metrics
    scores = np.array(scores)
    metrics = {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "score_std": float(np.std(scores)),
        "score_25th": float(np.percentile(scores, 25)),
        "score_75th": float(np.percentile(scores, 75)),
        "score_hist": [int(s) for s in scores],
    }

    # Reset agent to original state
    agent.epsilon = old_eps

    return metrics


def evaluate_episode(
    agent: YahtzeeAgent, env: YahtzeeEnv, encoder: StateEncoder
) -> Tuple[float, Dict[str, float]]:
    """
    Run a single evaluation episode and return the score and detailed metrics.
    """
    state = env.reset()
    done = False
    episode_reward = 0

    # Track detailed metrics
    metrics = {
        "rolls_used": 0,
        "upper_score": 0,
        "lower_score": 0,
        "yahtzees": 0,
        "bonus_achieved": 0,
    }

    while not done:
        state_vec = encoder.encode(state, opponent_value=0.5)
        valid_actions = env.get_valid_actions()
        action_idx = agent.select_action(state_vec, valid_actions)
        next_state, reward, done, info = env.step(action_idx)

        # Track metrics based on action
        if (
            info["action_type"] == "ROLL"
            or info["action_type"] == "HOLD"
        ):
            metrics["rolls_used"] += 1

        if info.get("category_scored"):
            category = info["category_scored"]
            points = info["points_scored"]

            # Track special combinations
            if category == "YAHTZEE" and points == 50:
                metrics["yahtzees"] += 1

            # Track upper/lower scores
            if category in [
                "ONES",
                "TWOS",
                "THREES",
                "FOURS",
                "FIVES",
                "SIXES",
            ]:
                metrics["upper_score"] += points
            else:
                metrics["lower_score"] += points

        episode_reward += reward
        state = next_state

    # Calculate final metrics
    upper_score = metrics["upper_score"]
    if upper_score >= 63:
        metrics["bonus_achieved"] = 1
        upper_score += 35

    # Final actual score (not reward-shaped)
    metrics["total_score"] = upper_score + metrics["lower_score"]

    return episode_reward, metrics


def get_latest_checkpoint(run_id: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint for the given run ID.
    """
    checkpoint_dir = os.path.join("models", run_id)
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth")
    ]
    if not checkpoints:
        return None

    return max(checkpoints, key=os.path.getmtime)


def load_checkpoint(
    checkpoint_path: str, agent: YahtzeeAgent, device: torch.device
) -> int:
    """
    Load agent state from a checkpoint file.
    Returns the episode number.
    """
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if it's a legacy format checkpoint that needs conversion
        if (
            not isinstance(checkpoint, dict)
            or "policy_net" not in checkpoint
        ):
            legacy_state = checkpoint
            agent.policy_net.load_state_dict(legacy_state)
            agent.target_net.load_state_dict(legacy_state)
            episode = 0
        else:
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
            episode = checkpoint.get("episode", 0)

        return episode
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0


def save_checkpoint(
    agent: YahtzeeAgent,
    episode: int,
    run_id: str,
    metrics: dict,
    is_best: bool = False,
) -> str:
    """
    Save a checkpoint of the agent's state.
    Returns the path to the saved checkpoint.
    """
    os.makedirs(os.path.join("models", run_id), exist_ok=True)

    # Only keep a limited number of checkpoints to save space
    checkpoint_dir = os.path.join("models", run_id)
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth") and not f.startswith("best_")
    ]

    # Keep only the 5 most recent checkpoints
    if len(checkpoints) >= 5:
        oldest_checkpoint = min(checkpoints, key=os.path.getmtime)
        os.remove(oldest_checkpoint)

    # Save current checkpoint
    checkpoint_path = os.path.join(
        "models", run_id, f"checkpoint_{episode:06d}.pth"
    )

    # Save only policy net weights and other essential info to save space
    checkpoint = {
        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "episode": episode,
        "learn_steps": agent.learn_steps,
        "metrics": metrics,
    }

    torch.save(checkpoint, checkpoint_path)

    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join("models", run_id, "best_model.pth")
        torch.save(checkpoint, best_path)

    return checkpoint_path


def train(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 100000,
    num_envs: int = 128,
    steps_per_update: int = 8,
    checkpoint_freq: int = 500,
    eval_freq: int = 100,
    num_eval_episodes: int = 50,
    min_improvement: float = 1.0,
    objective: str = "win",
    use_wandb: bool = True,
    wandb_project: str = "yahtzee-rl",
    use_enhanced_rewards: bool = True,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
    hidden_size: int = 128,
    use_noisy: bool = False,
    use_mixed_precision: bool = False,
) -> None:
    """
    Train the DQN agent using vectorized environments.
    """
    # Set up device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Enable cudnn benchmark for faster training on A100
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create run ID if not provided
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"yahtzee_{timestamp}"

    # Set up wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_id,
            config={
                "num_episodes": num_episodes,
                "num_envs": num_envs,
                "steps_per_update": steps_per_update,
                "checkpoint_freq": checkpoint_freq,
                "eval_freq": eval_freq,
                "num_eval_episodes": num_eval_episodes,
                "min_improvement": min_improvement,
                "objective": objective,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_enhanced_rewards": use_enhanced_rewards,
                "device": str(device),
            },
        )

    # Create environments
    reward_strategy = (
        RewardStrategy.STRATEGIC
        if use_enhanced_rewards
        else RewardStrategy.STANDARD
    )
    envs = [
        YahtzeeEnv(
            use_opponent_value=(objective == "win"),
            reward_strategy=reward_strategy,
        )
        for _ in range(num_envs)
    ]

    # Create encoder
    encoder = StateEncoder(use_opponent_value=(objective == "win"))

    # Create agent
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=46,  # Fixed size based on action space
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=0.997,
        target_update=50,
        device=str(device),
        hidden_size=hidden_size,
        use_noisy=use_noisy,
        use_amp=use_mixed_precision,
    )

    # Load checkpoint if provided
    start_episode = 0
    if checkpoint_path is not None:
        start_episode = load_checkpoint(
            checkpoint_path, agent, device
        )
    elif run_id is not None:
        latest_checkpoint = get_latest_checkpoint(run_id)
        if latest_checkpoint is not None:
            start_episode = load_checkpoint(
                latest_checkpoint, agent, device
            )

    # Initialize environment states
    states = [env.reset() for env in envs]

    # Tracking metrics
    best_score = 0
    scores_history = []
    learning_rates = []

    # Progress bar
    progress_bar = tqdm(
        range(start_episode, num_episodes),
        desc=f"Episode {start_episode}/{num_episodes}",
        unit="episode",
    )

    # Training loop
    for episode in progress_bar:
        episode_steps = 0
        episode_rewards = np.zeros(num_envs)

        # Episode loop
        while episode_steps < steps_per_update:
            # Encode states
            state_vecs = np.stack(
                [
                    encoder.encode(
                        s,
                        opponent_value=0.5
                        if objective == "win"
                        else 0.0,
                    )
                    for s in states
                ]
            )

            # Get actions
            valid_actions_list = [
                env.get_valid_actions() for env in envs
            ]
            actions = []
            for i in range(num_envs):
                actions.append(
                    agent.select_action(
                        state_vecs[i], valid_actions_list[i]
                    )
                )

            # Step environments
            next_states = []
            rewards = []
            dones = []

            for i, (env, action) in enumerate(zip(envs, actions)):
                next_state, reward, done, _ = env.step(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                episode_rewards[i] += reward

                # Reset if done
                if done:
                    next_states[i] = env.reset()

            # Store transitions and update agent
            loss = agent.train_step_batch(
                state_vecs,
                actions,
                rewards,
                [
                    encoder.encode(
                        s,
                        opponent_value=0.5
                        if objective == "win"
                        else 0.0,
                    )
                    for s in next_states
                ],
                dones,
            )

            # Update states
            states = next_states
            episode_steps += 1

        # Calculate metrics for logging
        mean_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)

        # Get current learning rate
        current_lr = agent.optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # Update progress bar
        metrics_dict = agent.get_metrics()
        progress_bar.set_description(
            f"Episode {episode}/{num_episodes} | "
            f"Reward: {mean_reward:.1f} | "
            f"Loss: {metrics_dict.get('loss', 0):.3f} | "
            f"LR: {current_lr:.1e}"
        )

        # Log to wandb
        if use_wandb:
            wandb_dict = {
                "episode": episode,
                "reward_mean": mean_reward,
                "reward_max": max_reward,
                "reward_min": min_reward,
                "epsilon": agent.epsilon,
                "learning_rate": current_lr,
            }
            wandb_dict.update(
                metrics_dict
            )  # Add all metrics from agent
            wandb.log(wandb_dict, step=episode)

        # Evaluation
        if episode > 0 and episode % eval_freq == 0:
            print(f"\nEvaluating agent at episode {episode}...")
            eval_metrics = evaluate_agent(
                agent,
                num_games=num_eval_episodes,
                objective=objective,
            )

            mean_score = eval_metrics["mean_score"]
            median_score = eval_metrics["median_score"]
            scores_history.append(mean_score)

            # Update best score and save model if improved
            if mean_score > best_score + min_improvement:
                best_score = mean_score
                print(
                    f"New best score: {best_score:.1f} (improved by {mean_score - best_score + min_improvement:.1f})"
                )
                save_checkpoint(
                    agent, episode, run_id, eval_metrics, is_best=True
                )

            # Display evaluation results
            print(f"Mean score: {mean_score:.1f}")
            print(f"Median score: {median_score:.1f}")
            print(f"Min score: {eval_metrics['min_score']:.1f}")
            print(f"Max score: {eval_metrics['max_score']:.1f}")

            # Log evaluation metrics to wandb
            if use_wandb:
                eval_dict = {
                    f"eval_{k}": v
                    for k, v in eval_metrics.items()
                    if k != "score_hist"
                }
                wandb.log(eval_dict, step=episode)

                # Log score distribution as histogram
                scores = eval_metrics["score_hist"]
                wandb.log(
                    {"eval_score_histogram": wandb.Histogram(scores)},
                    step=episode,
                )

        # Checkpoint saving
        if episode > 0 and episode % checkpoint_freq == 0:
            save_path = save_checkpoint(agent, episode, run_id, {})
            print(f"\nCheckpoint saved to {save_path}")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_agent(
        agent, num_games=100, objective=objective
    )

    # Save final model
    final_path = save_checkpoint(
        agent, num_episodes, run_id, final_metrics
    )
    print(f"Final model saved to {final_path}")

    # Print final results
    print("\nTraining completed!")
    print(f"Final mean score: {final_metrics['mean_score']:.1f}")
    print(f"Final median score: {final_metrics['median_score']:.1f}")

    # Close wandb
    if use_wandb:
        wandb.finish()

    # Plot training progress
    if len(scores_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(eval_freq, num_episodes + 1, eval_freq),
            scores_history,
        )
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Mean Score")
        plt.grid(True)
        plt.savefig(
            os.path.join("models", run_id, "training_progress.png")
        )
        plt.close()


def simulate_game(agent: YahtzeeAgent, render: bool = True) -> float:
    """Run a full game simulation and optionally render the state at each step."""
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=True)  # Match training
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    # Store original epsilon and set to minimum for deterministic play
    old_eps = agent.epsilon
    agent.epsilon = 0.02

    while not done:
        if render:
            print(f"\n=== Turn {turn} ===")
            print(env.render())

        state_vec = encoder.encode(state, opponent_value=0.5)
        valid_actions = env.get_valid_actions()
        action_idx = agent.select_action(state_vec, valid_actions)
        action = env.action_mapper.index_to_action(action_idx)

        state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        if render:
            if action.kind == ActionType.ROLL:
                print("Action: ROLL")
            elif action.kind == ActionType.HOLD:
                held = [
                    i + 1
                    for i, hold in enumerate(action.data)
                    if hold
                ]
                print(f"Action: HOLD {held}")
            else:
                print(
                    f"Action: SCORE {action.data.name} for {reward:.1f}"
                )

            if action.kind == ActionType.SCORE:
                turn += 1

    if render:
        print("\n=== Game Over ===")
        print(env.render())
        print(f"Final score: {total_reward:.1f}")

    # Restore original epsilon
    agent.epsilon = old_eps
    return total_reward


def show_action_values(
    agent: YahtzeeAgent,
    state: Optional[GameState] = None,
    num_top: int = 5,
) -> tuple:
    """Show expected values for all valid actions in the current state."""
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=True)

    if state is None:
        state = env.reset()

    print("\nCurrent Game State:")
    print(env.render())

    state_vec = encoder.encode(state, opponent_value=0.5)
    valid_actions = env.get_valid_actions()

    # Get Q-values and mask invalid actions
    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask

    # Sort actions by Q-value
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
            print(f"{i}. Hold {held} (EV: {value:.1f})")
        else:
            print(f"{i}. Score {action.data.name} (EV: {value:.1f})")

    return state, valid_q[:num_top]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Yahtzee RL Training Configuration"
    )

    # Core training parameters
    parser.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for this training run",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Number of episodes to train for",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=128,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training updates",
    )

    # Model parameters
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Size of hidden layers in the model",
    )
    parser.add_argument(
        "--use-noisy",
        action="store_true",
        help="Use noisy networks for exploration",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards",
    )
    parser.add_argument(
        "--steps-per-update",
        type=int,
        default=8,
        help="Environment steps before each training update",
    )

    # Reward and environment config
    parser.add_argument(
        "--objective",
        choices=["win", "score"],
        default="score",
        help="Training objective - win rate or maximum score",
    )

    # Evaluation and checkpoint settings
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100,
        help="Episodes between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of episodes for each evaluation",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=500,
        help="Episodes between checkpoints",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1.0,
        help="Minimum improvement in score to save best model",
    )

    # Wandb configuration
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="yahtzee-rl",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team or username)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=[],
        help="Tags for the W&B run",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    # Continue from checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to continue training from",
    )

    # Evaluation only mode
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on an existing model",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=1000,
        help="Number of games to evaluate for eval-only mode",
    )

    return parser.parse_args()


def save_config(args: argparse.Namespace, run_id: str) -> None:
    """Save training configuration to a file."""
    config_dir = os.path.join("models", run_id)
    os.makedirs(config_dir, exist_ok=True)

    # Convert args to dictionary
    config = vars(args)

    # Save config to JSON file
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {config_path}")


def create_run_id(args: argparse.Namespace) -> str:
    """Create a unique run ID if not provided."""
    if args.run_id:
        return args.run_id

    # Generate a unique identifier based on time and configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hash(f"{args.objective}_{args.hidden_size}")
    config_hash = (
        abs(config_hash) % 10000
    )  # Keep it reasonable length

    return f"yahtzee_{timestamp}_{config_hash}"


def run_evaluation_only(args: argparse.Namespace) -> None:
    """Run evaluation on an existing model without training."""
    if not args.checkpoint:
        raise ValueError(
            "--checkpoint must be specified for --eval-only mode"
        )

    print(f"Running evaluation only on checkpoint: {args.checkpoint}")

    # Set up device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create encoder
    encoder = StateEncoder(
        use_opponent_value=(args.objective == "win")
    )

    # Create agent
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=46,  # Fixed size based on action space
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=device,
        hidden_size=args.hidden_size,
        use_noisy=args.use_noisy,
        use_amp=args.use_mixed_precision,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    try:
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
    except:
        # Try legacy format
        print("Trying legacy format checkpoint...")
        agent.policy_net.load_state_dict(checkpoint)

    # Put agent in evaluation mode
    agent.epsilon = 0.01  # Use minimal epsilon for deterministic eval

    # Run evaluation
    print(f"Evaluating agent over {args.eval_games} games...")
    eval_metrics = evaluate_agent(
        agent, num_games=args.eval_games, objective=args.objective
    )

    # Print metrics
    print("\nEvaluation Results:")
    print(f"Mean Score: {eval_metrics['mean_score']:.1f}")
    print(f"Median Score: {eval_metrics['median_score']:.1f}")
    print(f"Min Score: {eval_metrics['min_score']:.1f}")
    print(f"Max Score: {eval_metrics['max_score']:.1f}")
    print(f"Score Std Dev: {eval_metrics['score_std']:.1f}")
    print(f"25th Percentile: {eval_metrics['score_25th']:.1f}")
    print(f"75th Percentile: {eval_metrics['score_75th']:.1f}")

    # Save detailed results
    results_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(
        results_dir, "evaluation_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Detailed results saved to {results_path}")


def main() -> None:
    """Main entry point for training configuration."""
    args = parse_args()

    # Create a unique run ID if not provided
    run_id = create_run_id(args)
    print(f"Run ID: {run_id}")

    # Save configuration
    save_config(args, run_id)

    # If evaluation only, run that and exit
    if args.eval_only:
        run_evaluation_only(args)
        return

    # Otherwise, run training
    train(
        run_id=run_id,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        num_envs=args.envs,
        steps_per_update=args.steps_per_update,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        num_eval_episodes=args.eval_episodes,
        min_improvement=args.min_improvement,
        objective=args.objective,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        use_noisy=args.use_noisy,
        use_mixed_precision=args.use_mixed_precision,
    )


if __name__ == "__main__":
    main()
