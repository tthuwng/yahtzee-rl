import numpy as np

from play_random import play_random_game


def run_random_games(n_games=1000):
    scores = []
    for i in range(n_games):
        score = play_random_game(verbose=False)
        scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} games...")

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    print(f"\nStatistics over {n_games} games:")
    print(f"Mean score: {mean_score:.1f}")
    print(f"Media   n score: {median_score:.1f}")
    print(f"Min score: {min_score}")
    print(f"Max score: {max_score}")

    return scores


if __name__ == "__main__":
    print("Playing one random game:")
    play_random_game(verbose=True)

    print("\nRunning 1000 games for statistics...")
    scores = run_random_games(1000)
