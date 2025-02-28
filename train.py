#!/usr/bin/env python3
import argparse
import datetime
import json
import os

import torch

from dqn import YahtzeeAgent
from encoder import StateEncoder
from main import evaluate_agent, train


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Yahtzee RL Training Configuration")

    # Core training parameters
    parser.add_argument(
        "--run-id", type=str, help="Unique identifier for this training run"
    )
    parser.add_argument(
        "--episodes", type=int, default=100000, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--envs", type=int, default=128, help="Number of parallel environments"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for training updates"
    )

    # Model parameters
    parser.add_argument(
        "--use-noisy",
        action="store_true",
        help="Use NoisyLinear layers for exploration",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="Size of hidden layers in the model",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of residual blocks in the model",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.997, help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--steps-per-update",
        type=int,
        default=8,
        help="Environment steps before each training update",
    )
    parser.add_argument(
        "--n-step", type=int, default=3, help="N-step returns for training"
    )

    # Reward and environment config
    parser.add_argument(
        "--use-enhanced-rewards",
        action="store_true",
        help="Use enhanced reward function",
    )
    parser.add_argument(
        "--objective",
        choices=["win", "score"],
        default="score",
        help="Training objective - win rate or maximum score",
    )

    # Evaluation and checkpoint settings
    parser.add_argument(
        "--eval-freq", type=int, default=100, help="Episodes between evaluations"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of episodes for each evaluation",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=500, help="Episodes between checkpoints"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1.0,
        help="Minimum improvement in score to save best model",
    )

    # A100 specific optimizations
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Use mixed precision training (FP16/BF16) for A100",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        help="Use CUDA graphs for faster training on A100",
    )

    # Wandb configuration
    parser.add_argument(
        "--wandb-project", type=str, default="yahtzee-rl", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity (team or username)"
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs="*", default=[], help="Tags for the W&B run"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )

    # Continue from checkpoint
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint to continue training from"
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


def setup_a100_optimizations(args: argparse.Namespace) -> None:
    """Set up CUDA configurations optimized for A100 GPUs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping A100 optimizations")
        return

    # Get device properties
    device_props = torch.cuda.get_device_properties(0)
    print(f"Using GPU: {device_props.name}")

    # Set up cudnn for faster training
    torch.backends.cudnn.benchmark = True

    # Set up mixed precision if requested
    if args.use_mixed_precision:
        # Set up for PyTorch native mixed precision
        # (The actual amp.autocast will be applied in training loop)
        print("Using mixed precision training")

        # Check if we can use bfloat16 (A100 supports this)
        if torch.cuda.is_bf16_supported():
            print("Using bfloat16 precision")
        else:
            print("Using float16 precision")

    # Set up CUDA graphs if requested
    if args.use_cuda_graphs:
        print("Using CUDA graphs for optimized training")
        # The actual CUDA graphs setup happens in the training loop

    # Set appropriate memory allocator
    torch.cuda.empty_cache()
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    # Print memory information
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")


def create_run_id(args: argparse.Namespace) -> str:
    """Create a unique run ID if not provided."""
    if args.run_id:
        return args.run_id

    # Generate a unique identifier based on time and configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hash(
        f"{args.objective}_{args.use_noisy}_{args.hidden_size}_{args.num_blocks}"
    )
    config_hash = abs(config_hash) % 10000  # Keep it reasonable length

    return f"yahtzee_{timestamp}_{config_hash}"


def run_evaluation_only(args: argparse.Namespace) -> None:
    """Run evaluation on an existing model without training."""
    if not args.checkpoint:
        raise ValueError("--checkpoint must be specified for --eval-only mode")

    print(f"Running evaluation only on checkpoint: {args.checkpoint}")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create encoder
    encoder = StateEncoder(use_opponent_value=(args.objective == "win"))

    # Create agent
    agent = YahtzeeAgent(
        state_size=encoder.state_size,
        action_size=46,  # Fixed size based on action space
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=device,
        use_noisy=args.use_noisy,
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
    results_path = os.path.join(results_dir, "evaluation_results.json")

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

    # Set up A100 optimizations if using CUDA
    setup_a100_optimizations(args)

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
        use_enhanced_rewards=args.use_enhanced_rewards,
        batch_size=args.batch_size,
        use_noisy=args.use_noisy,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
