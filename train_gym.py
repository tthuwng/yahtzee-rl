import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
from IPython.display import clear_output
from tqdm import tqdm

from dqn import YahtzeeAgent
from env import ActionType
from yahtzee_gym import RewardStrategy, YahtzeeGymEnv


def evaluate_episode(
    agent: YahtzeeAgent, env: YahtzeeGymEnv, objective: str = "win"
) -> float:
    """Run a single evaluation episode."""
    obs, _ = env.reset()
    total_reward = 0
    done = False

    old_eps = agent.epsilon
    agent.epsilon = 0.02

    while not done:
        valid_actions = env.env.get_valid_actions()
        if not valid_actions:
            break

        state_vec = torch.from_numpy(obs).float().to(agent.device)
        action_idx = agent.select_action(state_vec, valid_actions)
        obs, reward, done, truncated, info = env.step(action_idx)
        total_reward += reward

    agent.epsilon = old_eps
    return total_reward


def simulate_game(agent: YahtzeeAgent, render: bool = True) -> Tuple[float, float]:
    """Simulate a single game with visualization."""
    env = YahtzeeGymEnv(render_mode="human" if render else None)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    turn = 1

    old_eps = agent.epsilon
    agent.epsilon = 0.01

    while not done:
        if render:
            clear_output(wait=True)
            print(f"\n=== Turn {turn} | Score: {env.env.calc_score_total()} ===")
            env.render()

        valid_actions = env.env.get_valid_actions()
        if not valid_actions:
            break

        state_vec = torch.from_numpy(obs).float().to(agent.device)
        action_idx = agent.select_action(state_vec, valid_actions)
        action = env.env.idx_to_action[action_idx]

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
                points = env.env.calc_score(action.data, env.env.state.current_dice)
                print(f"Action: Score {action.data.name} for {points} points")

        obs, reward, done, truncated, info = env.step(action_idx)
        total_reward += reward

        if render and action.kind == ActionType.SCORE:
            print(f"Scored {info['base_score']} points")
            turn += 1

    final_score = info.get("final_score", 0)
    if render:
        clear_output(wait=True)
        print("\n=== Game Over ===")
        env.render()
        print(f"\nFinal Score: {final_score}")
        print(f"Total Reward: {total_reward:.1f}")

    agent.epsilon = old_eps
    return final_score, total_reward


def show_action_values(
    agent: YahtzeeAgent, env: Optional[YahtzeeGymEnv] = None, num_top: int = 5
) -> tuple:
    """
    Show expected values for all valid actions in the current state.
    If env is None, creates a new environment.
    Returns (env, valid_actions, q_values) for further use.
    """
    if env is None:
        env = YahtzeeGymEnv(render_mode="human")
        obs, _ = env.reset()
    else:
        obs = env.encoder.encode(env.env.state)

    print("\nCurrent Game State:")
    env.render()

    state_vec = torch.from_numpy(obs).float().to(agent.device)
    valid_actions = env.env.get_valid_actions()

    q_values = agent.get_q_values(state_vec)
    mask = np.full(agent.action_size, float("-inf"))
    mask[valid_actions] = 0
    q_values = q_values + mask

    valid_q = [(i, q_values[i]) for i in valid_actions]
    valid_q.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Actions and Their Expected Values:")
    for i, (action_idx, value) in enumerate(valid_q[:num_top], 1):
        action = env.env.idx_to_action[action_idx]
        if action.kind == ActionType.ROLL:
            print(f"{i}. ROLL all dice (EV: {value:.1f})")
        elif action.kind == ActionType.HOLD:
            held = [i + 1 for i, hold in enumerate(action.data) if hold]
            if held:
                print(f"{i}. Hold dice {held} (EV: {value:.1f})")
            else:
                print(f"{i}. ROLL all dice (EV: {value:.1f})")
        else:
            points = env.env.calc_score(action.data, env.env.state.current_dice)
            print(
                f"{i}. Score {action.data.name} for {points} points (EV: {value:.1f})"
            )

    return env, valid_q[:num_top]


def load_checkpoint(
    checkpoint_path: str, agent: YahtzeeAgent, device: torch.device
) -> int:
    """Load checkpoint and return starting episode."""
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


def train(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 100000,
    num_envs: int = 32,
    steps_per_update: int = 8,
    checkpoint_freq: int = 100,
    eval_freq: int = 50,
    num_eval_episodes: int = 50,
    min_improvement: float = 1.0,
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

    # Initialize environments
    reward_strat = (
        RewardStrategy.STRATEGIC if objective == "win" else RewardStrategy.STANDARD
    )
    envs = [
        YahtzeeGymEnv(
            use_opponent_value=(objective == "win"), reward_strategy=reward_strat
        )
        for _ in range(num_envs)
    ]
    # Separate env for evaluation
    eval_env = YahtzeeGymEnv(
        use_opponent_value=(objective == "win"), reward_strategy=reward_strat
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create agent
    agent = YahtzeeAgent(
        state_size=envs[0].observation_space.shape[0],
        action_size=envs[0].action_space.n,
        batch_size=1024,
        gamma=0.99,
        learning_rate=3e-4,
        target_update=50,
        device=device,
        min_epsilon=0.02,
        epsilon_decay=0.9997,
    )

    start_episode = 0
    metrics = {
        "best_eval_score": float("-inf"),
        "episode_rewards": [],
        "eval_scores": [],
        "losses": [],
    }

    if checkpoint_path:
        start_episode = load_checkpoint(checkpoint_path, agent, device)
        metrics.update(checkpoint.get("metrics", {}))

    last_save_time = time.time()
    progress = tqdm(range(start_episode, num_episodes), desc="Training")

    try:
        for episode in progress:
            episode_rewards = []
            episode_losses = []
            actual_scores = []

            # Reset all environments
            observations = [env.reset()[0] for env in envs]
            dones = [False] * num_envs
            total_rewards = [0.0] * num_envs

            # Run steps_per_update steps in all environments
            for _ in range(steps_per_update):
                # Skip done environments
                active_indices = [i for i, done in enumerate(dones) if not done]
                if not active_indices:
                    break

                # Prepare batch data
                active_obs = np.stack([observations[i] for i in active_indices])
                valid_actions_list = [
                    envs[i].env.get_valid_actions() for i in active_indices
                ]

                if not all(valid_actions_list):
                    continue

                # Select actions in batch
                actions = agent.select_actions_batch(active_obs, valid_actions_list)

                # Take actions and collect transitions
                next_observations = []
                rewards = []
                new_dones = []

                for idx, action in zip(active_indices, actions):
                    obs, reward, done, truncated, info = envs[idx].step(action)
                    next_observations.append(obs)
                    rewards.append(reward)
                    new_dones.append(done)
                    total_rewards[idx] += reward
                    observations[idx] = obs
                    dones[idx] = done

                    if done and "final_score" in info:
                        actual_scores.append(info["final_score"])

                # Train on batch of transitions
                if next_observations:
                    next_obs_batch = np.stack(next_observations)
                    loss = agent.train_step_batch(
                        active_obs,
                        actions,
                        rewards,
                        next_obs_batch,
                        new_dones,
                    )
                    if loss is not None:
                        episode_losses.append(loss)

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
                eval_stats = evaluate_agent(agent, eval_env, num_eval_episodes)
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
                    print(
                        f"\nNew best eval score: {mean_eval_score:.1f} "
                        f"(reward: {mean_eval_reward:.1f})"
                    )

            current_time = time.time()
            if current_time - last_save_time >= 1800:  # Save every 30 minutes
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
