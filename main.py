from datetime import datetime
from typing import Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from dqn import YahtzeeAgent
from encoder import StateEncoder
from env import IDX_TO_ACTION, NUM_ACTIONS, YahtzeeEnv


def simple_baseline_policy(state, env):
    """
    A very naive baseline that picks the category if it yields the highest immediate score,
    otherwise tries to roll for 'sixes'.
    Just for demonstration, you can replace with a more elaborate approach.
    """
    valid_actions = env.get_valid_actions()
    best_action = None
    best_score = -99999
    # Suppose we find if any SCORING action is valid with immediate high payoff
    for act_idx in valid_actions:
        act = IDX_TO_ACTION[act_idx]
        if act.kind.name == "SCORE":
            # Evaluate immediate reward for that category
            points = env.calc_score(act.data, env.state.current_dice)
            if points > best_score:
                best_score = points
                best_action = act_idx

    # If no better scoring found or we want to keep rolling
    if best_score < 15:  # small threshold
        # check if we can ROLL or HOLD
        roll_action = None
        hold_actions = []
        for a_idx in valid_actions:
            act = IDX_TO_ACTION[a_idx]
            if act.kind.name == "ROLL":
                roll_action = a_idx
            elif act.kind.name == "HOLD":
                hold_actions.append(a_idx)

        if roll_action is not None:
            return roll_action
        if hold_actions:
            # Just pick the hold that keeps 6's if present
            dice = env.state.current_dice
            mask = np.array([d == 6 for d in dice], dtype=bool)
            for h_idx in hold_actions:
                act = IDX_TO_ACTION[h_idx]
                if np.array_equal(act.data, mask):
                    return h_idx
            return hold_actions[0]

    return best_action


def populate_replay_with_baseline(agent: YahtzeeAgent, num_games: int = 50):
    """
    Runs a simple baseline for N full games, storing transitions in the agent's replay buffer.
    This 'pretraining' can help the agent see some decent transitions.
    """
    env = YahtzeeEnv()
    enc = StateEncoder()
    agent.eval()  # no epsilon exploration
    for _ in range(num_games):
        s = env.reset()
        done = False
        while not done:
            state_vec = enc.encode(s)
            a = simple_baseline_policy(s, env)
            next_s, r, done, _info = env.step(a)
            agent.store_transition(state_vec, a, r, enc.encode(next_s), done)
            s = next_s
    # Return to training mode
    agent.train()


def evaluate_agent(agent: YahtzeeAgent, num_games: int = 100) -> dict:
    """Evaluate agent performance across multiple full games."""
    env = YahtzeeEnv()
    encoder = StateEncoder(use_opponent_value=False)
    agent.eval()

    scores = []
    for _ in range(num_games):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            vec = encoder.encode(state)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = agent.select_action(vec, valid_actions)
            next_s, rew, done, _ = env.step(action)
            total_reward += rew
            state = next_s
        scores.append(total_reward)

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    mx = np.max(scores)
    mn = np.min(scores)
    print(
        f"Eval -> Mean: {mean_score:.1f}, Median: {median_score:.1f}, Max: {mx:.1f}, Min: {mn:.1f}"
    )
    agent.train()
    return {
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "max": mx,
        "min": mn,
    }


def train(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 150000,
    num_envs: int = 32,
    steps_per_update: int = 8,
    eval_freq: int = 200,
    num_eval_episodes: int = 50,
):
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="yahtzee-rl", name=f"yahtzee_run_{run_id}")

    # Create multiple envs
    envs = [YahtzeeEnv() for _ in range(num_envs)]
    encoders = [StateEncoder(use_opponent_value=False) for _ in range(num_envs)]

    # get state size
    dummy_s = envs[0].reset()
    dummy_vec = encoders[0].encode(dummy_s)
    state_size = len(dummy_vec)

    agent = YahtzeeAgent(
        state_size=state_size,
        action_size=NUM_ACTIONS,
        batch_size=512,
        gamma=0.99,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_step=3,
        target_update=100,
        min_epsilon=0.01,
        epsilon_decay=0.9995,
    )

    # Optional: load existing checkpoint
    if checkpoint_path:
        print("Loading checkpoint:", checkpoint_path)
        agent.load(checkpoint_path)

    # (1) Optional pretraining with baseline to populate replay
    print("Populating replay buffer with a simple baseline policy ...")
    populate_replay_with_baseline(agent, num_games=50)
    print("Baseline transitions added. Replay size:", len(agent.buffer))

    progress = tqdm(range(num_episodes), desc="Training")
    last_eval = 0

    # We'll do an epsilon scheduling example
    initial_epsilon = 1.0
    final_epsilon = agent.min_epsilon
    schedule_fraction = 0.5  # halfway
    for episode in progress:
        # Example of a custom scheduling approach
        fraction = min(1.0, episode / (num_episodes * schedule_fraction))
        agent.epsilon = initial_epsilon + fraction * (final_epsilon - initial_epsilon)

        # Reset each env
        states = [env.reset() for env in envs]
        dones = [False] * num_envs
        total_rewards = [0.0] * num_envs

        # do a short rollout for steps_per_update
        for _ in range(steps_per_update):
            active_idxs = [i for i, d in enumerate(dones) if not d]
            if not active_idxs:
                break

            # gather states
            state_vecs = []
            valid_actions_list = []
            for i in active_idxs:
                st_vec = encoders[i].encode(states[i])
                va = envs[i].get_valid_actions()
                state_vecs.append(st_vec)
                valid_actions_list.append(va)

            # select actions
            actions = []
            for svec, va in zip(state_vecs, valid_actions_list):
                if not va:
                    actions.append(None)
                else:
                    a = agent.select_action(svec, va)
                    actions.append(a)

            # step
            next_states = []
            rewards = []
            new_dones = []
            for idx, act in zip(active_idxs, actions):
                if act is None:
                    new_dones.append(True)
                    next_states.append(states[idx])
                    rewards.append(0.0)
                    dones[idx] = True
                else:
                    ns, rew, dn, _ = envs[idx].step(act)
                    next_states.append(ns)
                    rewards.append(rew)
                    new_dones.append(dn)
                    dones[idx] = dn
                    total_rewards[idx] += rew

            # train
            agent.train_step_batch(
                [states[i] for i in active_idxs],
                [a if a is not None else 0 for a in actions],
                rewards,
                [ns for ns in next_states],
                new_dones,
            )

            for i, s_new, d_new in zip(active_idxs, next_states, new_dones):
                states[i] = s_new
                dones[i] = d_new

        avg_reward = np.mean(total_rewards)
        wandb.log(
            {"episode": episode, "reward": avg_reward, "epsilon": agent.epsilon},
            step=episode,
        )
        progress.set_postfix({"MeanReward": f"{avg_reward:.1f}"})

        if episode - last_eval >= eval_freq:
            stats = evaluate_agent(agent, num_eval_episodes)
            wandb.log(
                {
                    "eval_mean": stats["mean"],
                    "eval_median": stats["median"],
                    "eval_std": stats["std"],
                },
                step=episode,
            )
            last_eval = episode

    # final: Save model with .pth extension so play.py and ui.py can pick it up
    save_path = f"models/yahtzee_run_{run_id}.pth"
    agent.save(save_path)
    print(f"Training complete. Model saved to {save_path}")
    wandb.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=150000)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    train(
        run_id=args.run_id,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        num_envs=args.num_envs,
    )


if __name__ == "__main__":
    main()
