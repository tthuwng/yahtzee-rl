import random

from env import Action, YahtzeeEnv


def play_random_game(verbose=True):
    env = YahtzeeEnv()
    state = env.reset()
    total_reward = 0
    done = False
    turn = 1

    while not done:
        if verbose:
            print(f"\n=== Turn {turn} ===")
            print(env.render())

        # First phase: rolling and holding dice
        while state.rolls_left > 0:
            valid_actions = env.get_valid_actions()
            # Filter out scoring actions during rolling phase
            roll_actions = [
                a
                for a in valid_actions
                if a == Action.ROLL or a in Action.get_hold_actions()
            ]
            action = random.choice(roll_actions)

            if verbose:
                print(f"\nChosen action: {action.name}")
            state, reward, done, _ = env.step(action)
            if verbose:
                print(env.render())

        # Second phase: must choose a scoring category
        valid_actions = env.get_valid_actions()
        # Only consider scoring actions
        score_actions = [a for a in valid_actions if a in Action.get_score_actions()]
        action = random.choice(score_actions)

        if verbose:
            print(f"\nScoring action chosen: {action.name}")
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if verbose:
            print(f"Scored {reward} points")

        turn += 1

    if verbose:
        print("\n=== Game Over ===")
        print(f"Final Score: {total_reward}")
    return total_reward


# Play a game
if __name__ == "__main__":
    score = play_random_game()
