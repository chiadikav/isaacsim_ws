# Franka Pick-and-Place with RL (Isaac Sim)

Training a Franka Emika Panda robot to pick a coloured block using reinforcement
learning (PPO) in NVIDIA Isaac Sim.

## Project Structure

```
franka_pick_and_place/
├── config.py          # All hyperparameters and paths (edit here first)
├── env.py             # PickPlaceEnv – Gymnasium environment (Isaac Sim backend)
├── train.py           # Training entry point with checkpointing
├── evaluate.py        # Evaluation script with JSON results output
├── analysis.ipynb     # Reward / success-rate plots (no Isaac Sim required)
├── model/             # Saved model checkpoints (.zip)
├── logs/              # TensorBoard logs + evaluation JSON
├── archive/           # Original monolithic files (kept for reference)
├── README.md          # This file
├── QUICKSTART.md      # Get running in 5 minutes
├── SETUP.md           # Full installation guide
├── COMPARISON.md      # MuJoCo vs Isaac Sim differences
└── MANIFEST.md        # Full file inventory
```

## Requirements

| Package | Version |
|---------|---------|
| NVIDIA Isaac Sim | 4.0+ |
| Python | 3.10 or 3.11 |
| gymnasium | 0.29.0+ |
| stable-baselines3 | 2.2.0+ |
| torch | 2.10.0+ (CUDA recommended) |
| numpy | 1.24.0+ |

## Quick Start

```bash
ISAAC_PYTHON=~/isaacsim/python.sh

# Train (100 k steps, ~2-3 min on a modern GPU)
$ISAAC_PYTHON train.py

# Evaluate and save results
$ISAAC_PYTHON evaluate.py --save-results logs/eval_results.json

# Visualise results (standard Python kernel – no Isaac Sim needed)
# Open analysis.ipynb in VS Code
```

See [QUICKSTART.md](QUICKSTART.md) for full usage and [SETUP.md](SETUP.md) for
installation.

## Training

PPO is used with a 256×256 MLP policy.

| Parameter | Value |
|-----------|-------|
| Action space | 8D (7 arm joints + gripper) |
| Observation space | 31D (arm state, EE pos, block pos/vel) |
| Max episode steps | 300 |
| Default timesteps | 100,000 |
| Checkpoint frequency | every 10,000 steps |

All values live in `config.py` — no need to touch `train.py` for routine changes.

## Model

After training, the final model is saved to `model/pick_place_ppo.zip`.
Intermediate checkpoints are saved every 10,000 steps as
`model/pick_place_ppo_<N>_steps.zip`.

```python
from stable_baselines3 import PPO
model = PPO.load("model/pick_place_ppo")
```

## Notes

- `train.py` and `evaluate.py` **must** be launched via Isaac Sim's `python.sh` —
  they cannot run under a standard Python interpreter.
- `analysis.ipynb` has no Isaac Sim dependency and runs under any Python 3 kernel.
- For MuJoCo ↔ Isaac Sim differences see [COMPARISON.md](COMPARISON.md).
