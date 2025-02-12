import os
import time
from datetime import datetime
from typing import Optional, Dict
import random

import numpy as np
import torch
import wandb
from IPython.display import clear_output
from tqdm import tqdm

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import IDX_TO_ACTION, NUM_ACTIONS, ActionType, GameState, YahtzeeEnv, YahtzeeCategory


def evaluate_agent(
    agent: YahtzeeAgent, num_games: int = 100, objective: str = "win"
) -> dict:
    """
    Evaluate agent performance across multiple games.
    Returns statistics for both training rewards and actual game scores.
    """
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=(objective == "win"))
    actual_scores = []
    training_rewards = []

    # Store original state and set to eval mode
    was_training = agent.training_mode
    agent.eval()

    # Run evaluation games
    for _ in range(num_games):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_vec = encoder.encode(
                state, opponent_value=0.5 if objective == "win" else 0.0
            )
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action_idx = agent.select_action(state_vec, valid_actions)
            state, reward, done, _ = env.step(action_idx)
            total_reward += reward

        # Track training reward
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

    # Restore original state
    if was_training:
        agent.train()

    # Calculate statistics for both metrics
    actual_scores = np.array(actual_scores)
    training_rewards = np.array(training_rewards)
    
    stats = {
        "actual_score": {
            "mean": np.mean(actual_scores),
            "median": np.median(actual_scores),
            "std": np.std(actual_scores),
            "max": np.max(actual_scores),
            "min": np.min(actual_scores),
        },
        "training_reward": {
            "mean": np.mean(training_rewards),
            "median": np.median(training_rewards),
            "std": np.std(training_rewards),
            "max": np.max(training_rewards),
            "min": np.min(training_rewards),
        }
    }

    print("\nEvaluation Results:")
    print("Actual Scores:")
    print(f"Mean: {stats['actual_score']['mean']:.1f} ± {stats['actual_score']['std']:.1f}")
    print(f"Max: {stats['actual_score']['max']:.1f}, Min: {stats['actual_score']['min']:.1f}")
    print("\nTraining Rewards:")
    print(f"Mean: {stats['training_reward']['mean']:.1f} ± {stats['training_reward']['std']:.1f}")
    print(f"Max: {stats['training_reward']['max']:.1f}, Min: {stats['training_reward']['min']:.1f}")
    
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
        score = int(metrics["eval_score"])
        filename = f"{models_dir}/yahtzee_run_{run_id}_best_eval{score}.pth"
    else:
        filename = f"{models_dir}/yahtzee_run_{run_id}_checkpoint_{episode}.pth"

    torch.save(checkpoint, filename)
    print(f"Saved model to: {filename}")
    return filename


def train(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 100000,
    num_envs: int = 32,
    steps_per_update: int = 8,
    checkpoint_freq: int = 100,
    eval_freq: int = 50,
    num_eval_episodes: int = 50,
    min_improvement: float = 1.0,  # Slightly larger gap to confirm improvement
    objective: str = "win",
) -> None:
    """Train DQN agent with parallel environments and batch processing."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        "num_episodes": num_episodes,
        "num_envs": num_envs,
        "steps_per_update": steps_per_update,
        "checkpoint_freq": checkpoint_freq,
        "eval_freq": eval_freq,
        "num_eval_episodes": num_eval_episodes,
        "min_improvement": min_improvement,
        "objective": objective,
    }

    wandb.init(
        project="yahtzee-rl",
        name=f"yahtzee_run_{run_id}",
        config=config,
        resume=checkpoint_path is not None,
    )

    # Initialize environments and encoders with appropriate objective
    envs = [YahtzeeEnv() for _ in range(num_envs)]
    encoders = [
        StateEncoder(use_opponent_value=(objective == "win")) for _ in range(num_envs)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create agent with optimized hyperparameters
    agent = YahtzeeAgent(
        state_size=encoders[0].state_size,
        action_size=NUM_ACTIONS,
        batch_size=1024,  # keep some parallelism, but not huge
        gamma=0.99,
        learning_rate=3e-4,  # keep a moderate LR
        target_update=50,
        device=device,
        min_epsilon=0.02,
        epsilon_decay=0.9997,  # slightly slower decay
    )

    start_episode = 0
    metrics = {
        "best_eval_score": float("-inf"),
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

    try:
        for episode in progress:
            episode_rewards = []
            episode_losses = []
            actual_scores = []  # Track actual scores for each environment

            # Reset all environments
            states = [env.reset() for env in envs]
            dones = [False] * num_envs
            total_rewards = [0.0] * num_envs

            # Run steps_per_update steps in all environments
            for _ in range(steps_per_update):
                # Skip done environments
                active_indices = [i for i, done in enumerate(dones) if not done]
                if not active_indices:
                    break

                # Prepare batch data
                active_states = [states[i] for i in active_indices]
                state_vecs = np.stack(
                    [
                        encoders[i].encode(
                            s, opponent_value=0.5 if objective == "win" else 0.0
                        )
                        for i, s in zip(active_indices, active_states)
                    ]
                )
                valid_actions_list = [
                    envs[i].get_valid_actions() for i in active_indices
                ]

                if not all(valid_actions_list):
                    continue

                # Select actions in batch
                actions = agent.select_actions_batch(state_vecs, valid_actions_list)

                # Take actions and collect transitions
                next_states = []
                rewards = []
                new_dones = []

                for idx, action in zip(active_indices, actions):
                    next_state, reward, done, _ = envs[idx].step(action)
                    next_states.append(next_state)
                    rewards.append(reward)
                    new_dones.append(done)
                    total_rewards[idx] += reward
                    states[idx] = next_state
                    dones[idx] = done

                # Train on batch of transitions
                if next_states:
                    next_state_vecs = np.stack(
                        [
                            encoders[i].encode(
                                s, opponent_value=0.5 if objective == "win" else 0.0
                            )
                            for i, s in zip(active_indices, next_states)
                        ]
                    )
                    loss = agent.train_step_batch(
                        state_vecs,
                        actions,
                        rewards,
                        next_state_vecs,
                        new_dones,
                    )
                    if loss is not None:
                        episode_losses.append(loss)

            # Calculate actual scores for completed environments
            for i, (state, done) in enumerate(zip(states, dones)):
                if done:
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

            episode_rewards.extend(total_rewards)
            metrics["episode_rewards"].extend(episode_rewards)
            if episode_losses:
                metrics["losses"].extend(episode_losses)

            mean_reward = np.mean(episode_rewards)
            mean_loss = np.mean(episode_losses) if episode_losses else None
            mean_actual_score = np.mean(actual_scores) if actual_scores else None

            wandb.log(
                {
                    "episode": episode,
                    "reward": mean_reward,
                    "actual_score": mean_actual_score,
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
                progress.set_postfix(
                    {
                        "Mean100": f"{mean_100:.1f}",
                        "Best": f"{best_score:.1f}",
                    }
                )

            # Evaluate agent
            if (episode + 1) % eval_freq == 0:
                eval_stats = evaluate_agent(agent, num_eval_episodes, objective)
                mean_eval_score = eval_stats["actual_score"]["mean"]
                mean_eval_reward = eval_stats["training_reward"]["mean"]
                metrics["eval_scores"].append(mean_eval_score)
                metrics["eval_score"] = mean_eval_score

                wandb.log(
                    {
                        "eval_score": mean_eval_score,
                        "eval_score_std": eval_stats["actual_score"]["std"],
                        "eval_reward": mean_eval_reward,
                        "eval_reward_std": eval_stats["training_reward"]["std"],
                    },
                    step=episode,
                )

                if mean_eval_score > metrics["best_eval_score"] + min_improvement:
                    metrics["best_eval_score"] = mean_eval_score
                    filename = save_checkpoint(agent, episode, run_id, metrics, True)
                    wandb.save(filename)
                    print(f"\nNew best eval score: {mean_eval_score:.1f} (reward: {mean_eval_reward:.1f})")

            current_time = time.time()
            if current_time - last_save_time >= 1800:
                filename = save_checkpoint(agent, episode + 1, run_id, metrics)
                last_save_time = current_time

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        try:
            print("\nSaving final model...")
            final_metrics = {
                **metrics,
                "final_mean_reward": np.mean(metrics["episode_rewards"][-100:]),
            }
            filename = save_checkpoint(agent, episode + 1, run_id, final_metrics)
            print(f"Final model saved to: {filename}")
        except Exception as e:
            print(f"Error saving final model: {str(e)}")

        if wandb.run is not None:
            wandb.finish()


def evaluate_episode(
    agent: YahtzeeAgent, env: YahtzeeEnv, encoder: StateEncoder
) -> float:
    """Run a single evaluation episode."""
    state = env.reset()
    total_reward = 0
    done = False

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

    agent.epsilon = old_eps
    return total_reward


def simulate_game(agent: YahtzeeAgent, render: bool = True) -> float:
    """Simulate a single game with visualization."""
    env = YahtzeeEnv()
    encoder = StateEncoder()
    state = env.reset()
    done = False
    turn = 1

    old_eps = agent.epsilon
    agent.epsilon = 0.01

    while not done:
        if render:
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
            print(f"\n=== Turn {turn} | Score: {actual_score} ===")
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
            elif action.kind == ActionType.HOLD:
                held = [i + 1 for i, hold in enumerate(action.data) if hold]
                if held:
                    print(f"Action: Hold dice at positions {held}")
                else:
                    print("Action: ROLL all dice")
            else:
                points = env.calc_score(action.data, state.current_dice)
                print(f"Action: Score {action.data.name} for {points} points")

        state, reward, done, _ = env.step(action_idx)
        if render and action.kind == ActionType.SCORE:
            points = env.calc_score(action.data, state.current_dice)
            print(f"Scored {points} points")
            turn += 1

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

    if render:
        clear_output(wait=True)
        print("\n=== Game Over ===")
        print(env.render())
        print(f"\nFinal Score: {actual_score}")
        print(f"• Upper Section: {upper_total}")
        print(f"• Upper Bonus: {bonus}")
        print(f"• Lower Section: {lower_total}")

    agent.epsilon = old_eps
    return actual_score, reward


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

    state_vec = encoder.encode(state)
    valid_actions = env.get_valid_actions()

    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask

    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Actions and Their Expected Values:")
    for i, (action_idx, value) in enumerate(valid_q[:num_top], 1):
        action = env.IDX_TO_ACTION[action_idx]
        if action.kind == ActionType.ROLL:
            print(f"{i}. ROLL all dice (EV: {value:.1f})")
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                print(f"{i}. Hold dice {held} (EV: {value:.1f})")
            else:
                print(f"{i}. ROLL all dice (EV: {value:.1f})")
        else:
            points = env.calc_score(action.data, state.current_dice)
            print(f"{i}. Score {action.data.name} for {points} points (EV: {value:.1f})")

    return state, valid_q[:num_top]


def main():
    """Main training function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Yahtzee DQN agent")
    parser.add_argument("--run_id", type=str, help="Run ID for resuming training")
    parser.add_argument(
        "--episodes", type=int, default=50000, help="Number of episodes"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--num_envs", type=int, default=32, help="Number of parallel environments"
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="win",
        choices=["win", "avg_score"],
        help="Training objective (win or avg_score)",
    )
    args = parser.parse_args()

    train(
        run_id=args.run_id,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        num_envs=args.num_envs,
        objective=args.objective,
    )


if __name__ == "__main__":
    main()