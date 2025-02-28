# Yahtzee Reinforcement Learning

This project implements a Deep Q-Learning agent for the game of Yahtzee, optimized for high-performance training on A100 GPUs.

## Features

- Deep Q-Network (DQN) with dueling architecture and noisy networks for efficient exploration
- Enhanced state representation with strategic game features
- Sophisticated reward function with game-phase aware incentives
- Optimized for A100 GPU training:
  - Mixed precision (FP16/BF16) support
  - Efficient memory management with checkpointing
  - Weights & Biases integration for experiment tracking
- Evaluation tools to benchmark agent performance

## Requirements

```
torch>=2.0.0
numpy>=1.20.0
matplotlib>=3.3.0
wandb>=0.12.0
tqdm>=4.60.0
```

## Getting Started

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/yahtzee-rl.git
cd yahtzee-rl
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Training

To train a model with default parameters optimized for A100 GPUs:

```bash
python train.py --use-noisy --use-enhanced-rewards --use-mixed-precision
```

For a full training run to achieve 230+ median/240+ mean scores:

```bash
python train.py \
  --episodes 200000 \
  --envs 256 \
  --batch-size 4096 \
  --use-noisy \
  --hidden-size 512 \
  --num-blocks 4 \
  --learning-rate 1e-4 \
  --gamma 0.997 \
  --steps-per-update 8 \
  --n-step 3 \
  --use-enhanced-rewards \
  --objective score \
  --use-mixed-precision
```

### Training with Weights & Biases

To track your experiments with W&B:

```bash
python train.py \
  --wandb-project yahtzee-rl \
  --wandb-entity your-username \
  --wandb-tags a100 high-score \
  --use-noisy \
  --use-enhanced-rewards \
  --use-mixed-precision
```

### Evaluation

To evaluate a trained model:

```bash
python train.py --eval-only --checkpoint models/your_run_id/best_model.pth --eval-games 1000
```

## Project Structure

- `dqn.py`: Implementation of the Deep Q-Network and agent
- `encoder.py`: State representation and encoding
- `env.py`: Yahtzee game environment
- `main.py`: Main training loop and utilities
- `train.py`: Training configuration and entry point
- `play.py`: Interactive gameplay and model visualization
- `yahtzee_types.py`: Type definitions for the Yahtzee game

## A100 Optimization Guide

This codebase is optimized for training on NVIDIA A100 GPUs. Key optimization features include:

1. **Mixed Precision Training**: Uses FP16/BF16 operations where appropriate to increase throughput
2. **Memory Efficiency**: Careful batch sizes and memory management
3. **Checkpoint Management**: Automatically prunes older checkpoints to save storage
4. **Vectorized Environment**: Processes multiple game environments in parallel

### Performance Expectations

When training on an A100 40GB GPU with recommended settings:

- Expected training time: ~24-48 hours to reach 230+ median score
- Memory usage: ~20-25GB
- Expected results: 230+ median score, 240+ mean score

### Troubleshooting A100 Issues

If you encounter CUDA out-of-memory errors:

- Reduce `--batch-size` to 2048 or 1024
- Reduce `--envs` to 128 or 64

If performance is slower than expected:

- Ensure `--use-mixed-precision` is enabled
- Monitor GPU utilization with `nvidia-smi`
- Try adjusting `--steps-per-update`

## Achieving High Scores

To achieve the target scores of 230+ median and 240+ mean:

1. Use the enhanced rewards (`--use-enhanced-rewards`)
2. Enable noisy networks for better exploration (`--use-noisy`)
3. Train for at least 150,000 episodes
4. Use a large replay buffer (default is 200,000)
5. Train with a discount factor of 0.997 or higher

The most important factors for high scores are:

1. Strategic state representation (already implemented)
2. Enhanced reward shaping (already implemented)
3. Sufficient training time
4. Exploration-exploitation balance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Baselines](https://github.com/openai/baselines) for DQN implementation inspiration
- [Weights & Biases](https://wandb.ai/) for experiment tracking
