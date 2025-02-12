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


def evaluate_agent(
    agent: YahtzeeAgent, num_games: int = 100, objective: str = "win"
) -> dict:
    """Evaluate agent performance across multiple games."""
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=(objective == "win"))
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
            state_vec = encoder.encode(
                state, opponent_value=0.5 if objective == "win" else 0.0
            )
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
    """
    Extra shaping to encourage high-value categories, upper bonus, etc.
    Adjust as needed for better policies.
    """
    base_reward = score
    bonus_reward = 2.0

    dice = env.state.current_dice
    counts = np.bincount(dice, minlength=7)[1:] if np.any(dice) else np.array([0] * 6)
    max_count = int(max(counts)) if counts.size > 0 else 0
    unique_vals = np.unique(dice) if np.any(dice) else np.array([])

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
    num_scored = sum(1 for s in env.state.score_sheet.values() if s is not None)
    is_early_game = num_scored < 6

    # Encourage high-of-a-kind & Yahtzees
    if max_count >= 4:
        bonus_reward += 5.0
    elif max_count == 3:
        bonus_reward += 2.0

    # Encourage upper bonus
    if upper_remaining > 0:
        points_needed = max(0, 63 - upper_score)
        avg_needed = points_needed / upper_remaining if upper_remaining > 0 else 0
        if category in upper_cats:
            val = upper_cats.index(category) + 1
            if score >= val * 3:
                bonus_reward += 8.0
            elif score >= avg_needed:
                bonus_reward += 3.0
            elif score > 0:
                bonus_reward += 1.0
            bonus_reward += val * 0.2

    # Straights
    if len(unique_vals) >= 4:
        sorted_vals = np.sort(unique_vals)
        gaps = np.diff(sorted_vals)
        if np.all(gaps == 1):
            bonus_reward += 5.0

    # If we got 0 points
    if score == 0:
        bonus_reward -= 5.0

    # Big combos
    if category == YahtzeeCategory.YAHTZEE and score == 50:
        bonus_reward += 15.0
    elif category == YahtzeeCategory.LARGE_STRAIGHT and score == 40:
        bonus_reward += 10.0
    elif category == YahtzeeCategory.SMALL_STRAIGHT and score == 30:
        bonus_reward += 7.0
    elif category == YahtzeeCategory.FULL_HOUSE and score == 25:
        bonus_reward += 5.0

    # Late game slight bonus
    if num_scored >= 10 and score > 0:
        bonus_reward += 2.0

    final_reward = base_reward + bonus_reward
    return final_reward


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
        # Always use eval_score for best checkpoints
        score = int(metrics["eval_score"])
        filename = f"{models_dir}/yahtzee_run_{run_id}_best_eval{score}.pth"
    else:
        # Regular checkpoints include episode number
        filename = f"{models_dir}/yahtzee_run_{run_id}_checkpoint_{episode}.pth"

    torch.save(checkpoint, filename)
    print(f"Saved model to: {filename}")
    return filename


def train(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 50000,
    num_envs: int = 32,
    steps_per_update: int = 8,
    checkpoint_freq: int = 100,
    eval_freq: int = 50,
    num_eval_episodes: int = 50,
    min_improvement: float = 0.5,
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

    envs = [YahtzeeEnv() for _ in range(num_envs)]
    encoders = [
        StateEncoder(use_opponent_value=(objective == "win")) for _ in range(num_envs)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = YahtzeeAgent(
        state_size=encoders[0].state_size,
        action_size=NUM_ACTIONS,
        batch_size=4096,
        gamma=0.999,
        learning_rate=2.5e-4,
        target_update=500,
        device=device,
        min_epsilon=0.02,
        epsilon_decay=0.9995,
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
            states = [env.reset() for env in envs]
            dones = [False] * num_envs
            total_rewards = [0.0] * num_envs

            for _ in range(steps_per_update):
                active_indices = [i for i, done in enumerate(dones) if not done]
                if not active_indices:
                    break

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

                actions = agent.select_actions_batch(state_vecs, valid_actions_list)

                next_states = []
                rewards = []
                new_dones = []
                for idx, action in zip(active_indices, actions):
                    next_state, reward, done, info = envs[idx].step(action)

                    if (
                        info.get("action_type") == "SCORE"
                        and "category_scored" in info
                        and "points_scored" in info
                    ):
                        shaped = calculate_strategic_reward(
                            envs[idx],
                            YahtzeeCategory[info["category_scored"]],
                            info["points_scored"],
                        )
                        reward = shaped

                    next_states.append(next_state)
                    rewards.append(reward)
                    new_dones.append(done)
                    total_rewards[idx] += reward
                    states[idx] = next_state
                    dones[idx] = done

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
                        active_indices,
                    )
                    if loss is not None:
                        episode_losses.append(loss)

            episode_rewards.extend(total_rewards)
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

            if len(metrics["episode_rewards"]) >= 100:
                mean_100 = np.mean(metrics["episode_rewards"][-100:])
                best_score = metrics["best_eval_score"]
                progress.set_postfix(
                    {
                        "Mean100": f"{mean_100:.1f}",
                        "Best": f"{best_score:.1f}",
                    }
                )

            if (episode + 1) % eval_freq == 0:
                eval_stats = evaluate_agent(agent, num_eval_episodes, objective)
                mean_eval_score = eval_stats["mean"]
                metrics["eval_scores"].append(mean_eval_score)
                metrics["eval_score"] = mean_eval_score

                wandb.log(
                    {
                        "eval_score": mean_eval_score,
                        "eval_score_std": eval_stats["std"],
                    },
                    step=episode,
                )

                if mean_eval_score > metrics["best_eval_score"] + min_improvement:
                    metrics["best_eval_score"] = mean_eval_score
                    filename = save_checkpoint(agent, episode, run_id, metrics, True)
                    wandb.save(filename)
                    print(f"\nNew best eval score: {mean_eval_score:.1f}")

            current_time = time.time()
            # Periodically checkpoint regardless of eval improvements
            if current_time - last_save_time >= 1800:
                filename = save_checkpoint(agent, episode + 1, run_id, metrics)
                wandb.save(filename)
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
                "final_mean_reward": np.mean(metrics["episode_rewards"][-100:])
                if metrics["episode_rewards"]
                else 0,
            }
            filename = save_checkpoint(agent, episode + 1, run_id, final_metrics)
            wandb.save(filename)
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
            elif action.kind == ActionType.HOLD:
                held = [i + 1 for i, hold in enumerate(action.data) if hold]
                print(f"Action: Hold dice at positions {held}")
            else:
                print(f"Action: Score {action.data.name}")

        state, reward, done, info = env.step(action_idx)
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
            print(f"{i + 1}. ROLL all dice (EV: {value:.1f})")
        elif action.kind == ActionType.HOLD:
            held = [j + 1 for j, hold in enumerate(action.data) if hold]
            if held:
                print(f"{i + 1}. Hold dice {held} (EV: {value:.1f})")
            else:
                print(f"{i + 1}. ROLL all dice (EV: {value:.1f})")
        else:
            print(f"{i + 1}. Score {action.data.name} (EV: {value:.1f})")

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
        "--num_envs", type=int, default=16, help="Number of parallel environments"
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
