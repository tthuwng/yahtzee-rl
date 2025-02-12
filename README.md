### Overview

**Yahtzee RL Challenge**. This repository contains a Deep Q-Learning (DQN) agent that learns to play the dice game Yahtzee. The main goals of this project are:
1. **Simulation Mode**: Observe the AI agent playing a complete Yahtzee game autonomously.
2. **Calculation Mode**: Explore individual states, see which actions are possible, and discover the agent’s predicted expected values for each action.

We aim for the agent to achieve high average scores (ideally around 200–240) by effectively balancing immediate and long-term strategies.

### Project Layout

- **`env.py`**: Contains the `YahtzeeEnv` class, a Gym-like environment simulating Yahtzee mechanics (dice rolling, scoring, etc.).
- **`encoder.py`**: Defines the `StateEncoder` that converts a game state (dice values, scores, etc.) into a numerical vector suitable for the neural network.
- **`dqn.py`**: Implements the Deep Q-Network (`DQN`) architecture, the `YahtzeeAgent`, and the replay buffer.
- **`play.py`**: Utility functions for simulating games, interacting with the agent, and evaluating performance.
- **`app.py`**: A Gradio-based UI that allows you to load models, watch simulations, and examine agent decisions step by step.
- **`main.py`**: A script for training the agent with support for logging, checkpoints, and optional W&B (Weights & Biases) integration for experiment tracking.

### Installation & Usage

1. **Install Dependencies**  
   Make sure you have Python 3.9+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Training the Agent**  
   You can launch the training process via:
   ```bash
   python main.py --episodes 50000 --num_envs 32 --objective avg_score
   ```
   - `--episodes`: number of training episodes.
   - `--num_envs`: number of parallel environments for faster training.
   - `--objective`: `"win"` (the agent tries to outscore a hypothetical opponent) or `"avg_score"` (purely maximize final score).

3. **Launching the Gradio UI**  
   After training, start the UI by running:
   ```bash
   python app.py
   ```
   This will launch a local server (e.g. at `http://127.0.0.1:7860`), providing:
   - A **Model dropdown** to load a `.pth` model file from the `models/` directory.
   - An **Objective radio** to choose between `"win"` and `"avg_score"`.
   - **Simulation Tab**: Run a full game simulation.
   - **Calculation Tab**: Step by step game analysis of the best actions.
   - **Analysis Tab**: Evaluate the performance of the loaded model over multiple games (e.g. 100).

### Architecture & Hyperparameter Decisions

1. **DQN Architecture**  
   - **Input**: Encoded game state of size ~22–23 floats (dice counts, rolls left, score categories filled, etc.).
   - **Hidden Layers**: A series of fully connected layers (e.g., size 512), often with *residual blocks* for improved gradient flow.
   - **Dueling Network**: Splits the network into value and advantage streams before recombining to get Q-values.
   - **Output**: Q-values for each possible Yahtzee action (e.g., rolling, holding, scoring various categories).

2. **Hyperparameters**  
   - **Replay Buffer**: Prioritized replay, capacity ~200k transitions, alpha=0.7, beta=0.5
   - **Batch Size**: ~1024 or 2048
   - **Learning Rate**: 5e-5 or 2e-4 in different runs
   - **Gamma**: 0.99 or 0.997 for discount factor
   - **\(\epsilon\)-greedy**: Starts at 1.0 and decays to ~0.02
   - **Target Network Update**: Soft update or periodically (e.g. every 50 steps)
   - **Reward Shaping**:
     - Basic approach: immediate points + potential bonus for certain combos.
     - Strategic approach: more reward for bigger combos, bonus for avoiding 0-scores, etc.
     - Potential-based approach: shaping to encourage progress toward 63 in the upper section, etc.

3. **Training Strategy**  
   - Running multiple parallel environments to collect experience quickly.
   - Periodically evaluating on fixed seeds or sets of episodes to track actual final scores (the “real” objective).
   - Using either `avg_score` objective for pure scoring or `win` objective to consider an “opponent’s value” in encoding.

### Results & Observations

- **Current Performance**:  
  - Mean scores around 120–140 after ~15k–30k steps with certain reward shaping strategies.
  - With more training and refined hyperparameters, the agent can sometimes reach ~180–200 in certain runs, but it requires longer training (e.g., 50k+ episodes) and more careful hyperparameter tuning.
- **Learning Curve**:  
  - Often, the agent’s “training reward” might differ from the actual final “Yahtzee score.” This discrepancy is because the model might maximize intermediate shaped rewards rather than raw final game points. Adjusting or aligning these metrics is crucial to converge toward a truly high final Yahtzee score.
- **Key Observations**:
  1. **Reward Shaping** drastically influences how quickly or even if the agent converges to a strong strategy.
  2. The large action space (rolling vs. holding patterns) can make it tricky for the agent to explore effectively, requiring robust exploration strategies.
  3. Debugging needed to confirm no misalignment between “reward” vs. actual “final score.”

### Lessons Learned

- **Reward vs. Final Score**: Always verify that your shaping signals do not overshadow the real objective. Over-shaping can lead to the agent optimizing extrinsic signals (like small incremental rewards) rather than final game score.
- **Exploration**: Epsilon schedules matter a lot. The agent might get stuck in suboptimal patterns if \(\epsilon\) decays too fast.
- **Stability**: Large neural nets (512 units + residual blocks) need stable training techniques (e.g., gradient clipping, careful learning rates).
- **Compute**: Achieving 200+ median scores often demands **longer training** (on the order of tens of thousands of episodes) plus thorough hyperparameter searching.

### Potential Next Steps

- **Longer Training**: Extending episodes to ~50k or more while monitoring final score (not just shaped reward).
- **Better Reward Shaping** or even **no shaping** with direct final reward to reduce confusion between actual final scoring and partial steps.
- **Self-play or Opponent**: If focusing on “win” mode, introduce a strong reference opponent or advanced simulation strategies.

### References
- [Yahtzee RL Example (Stanford PDF)](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf)
- [Yahtzotron: Advantage Actor-Critic Approach](https://dionhaefner.github.io/2021/04/yahtzotron-learning-to-play-yahtzee-with-advantage-actor-critic/)
- [Markus Dutschke’s Implementation](https://github.com/markusdutschke/yahtzee)
- [Yahtzee RL Articles](https://www.yahtzeemanifesto.com/reinforcement-learning-yahtzee.php)
