import numpy as np
import torch

from dqn import YahtzeeAgent
from yahtzee_gym import YahtzeeGymEnv


def test_random_game():
    """Test playing a random game with the Gym environment."""
    env = YahtzeeGymEnv(render_mode="human")
    obs, _ = env.reset()
    done = False
    total_reward = 0

    print("\nStarting random game...\n")
    while not done:
        # Sample random valid action
        valid_actions = (
            env.env.get_valid_actions()
        )  # Access underlying env for valid actions
        action = np.random.choice(valid_actions)

        # Take step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Render
        env.render()
        print("-" * 40)

    print(f"\nGame Over! Total reward: {total_reward:.1f}")
    print(f"Final score: {info.get('final_score', 0)}")


def test_trained_agent():
    """Test playing a game with a trained agent."""
    # Initialize environment
    env = YahtzeeGymEnv(use_opponent_value=True, render_mode="human")

    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = YahtzeeAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=512,
        gamma=0.99,
        learning_rate=3e-4,
        target_update=50,
        device=device,
    )

    # Load trained weights if available
    try:
        agent.load("models/best_model.pth")  # Update path as needed
        print("\nLoaded trained agent")
    except FileNotFoundError:
        print("\nNo trained model found, using random agent")

    # Play game
    obs, _ = env.reset()
    done = False
    total_reward = 0

    print("\nStarting game with trained agent...\n")
    while not done:
        # Get valid actions
        valid_actions = env.env.get_valid_actions()

        # Get action from agent
        state_vec = torch.from_numpy(obs).float().to(device)
        action = agent.select_action(state_vec, valid_actions)

        # Take step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Render
        env.render()
        print("-" * 40)

    print(f"\nGame Over! Total reward: {total_reward:.1f}")
    print(f"Final score: {info.get('final_score', 0)}")


if __name__ == "__main__":
    # Test random gameplay
    test_random_game()

    print("\n" + "=" * 80 + "\n")

    # Test with trained agent
    test_trained_agent()
