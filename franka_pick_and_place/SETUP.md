# Setup Guide

Complete installation and configuration guide for Franka Pick-and-Place (Isaac Sim).

## System Prerequisites

| Requirement | Details |
|-------------|----------|
| OS | Ubuntu 20.04 or 22.04 (recommended) |
| GPU | NVIDIA RTX 3090 / RTX 4090 / A100 or better |
| NVIDIA Driver | 550.x or later (`nvidia-smi` to verify) |
| Python | 3.10 or 3.11 |
| Disk space | ~20 GB for Isaac Sim |

## Step 1 – Install NVIDIA Isaac Sim

**Option A – Direct download (recommended)**

```bash
# Visit https://developer.nvidia.com/isaac-sim and download version 4.0+
tar -xzf isaac-sim-*.tar.gz -C ~/.local/share/ov/pkg/
```

**Option B – Omniverse Launcher**

1. Download the [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/download/)
2. Sign in with a free NVIDIA developer account
3. Search for **Isaac Sim**, select version 4.0+, and click **Install**

## Step 2 – Initialise the Isaac Sim Environment

```bash
source ~/isaacsim/setup_python_env.sh

# Verify
python -c "from isaacsim import SimulationApp; print('✓ Isaac Sim ready')"
```

For convenience, add an alias to your shell profile:

```bash
echo "alias isaac='source ~/isaacsim/setup_python_env.sh'" >> ~/.bashrc
```

## Step 3 – Install Python Dependencies

```bash
conda activate isaac-sim   # or whichever env Isaac Sim created
pip install gymnasium stable-baselines3 torch

# Optional: for analysis.ipynb
pip install matplotlib pandas tensorboard
```

## Step 4 – Verify the Full Stack

```bash
python -c "
import torch, gymnasium, stable_baselines3, numpy
print('✓ All packages present')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"
```

## Step 5 – Run a Smoke Test

```bash
ISAAC_PYTHON=~/isaacsim/python.sh
$ISAAC_PYTHON train.py --timesteps 1000
```

Expected output:
```
✓ SimulationApp started (headless=False)
✓ Franka loaded from: ...
✓ Target block added (DynamicCuboid)
Using cuda device
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 300      |
|    ep_rew_mean      | ...      |
```

> The Omniverse viewport will open. Pass `--headless` to suppress it.

## Project Layout

```
franka_pick_and_place/
├── config.py          ← edit hyperparams here
├── env.py             ← Gymnasium environment
├── train.py           ← run with python.sh to train
├── evaluate.py        ← run with python.sh to evaluate
├── analysis.ipynb     ← open in VS Code to plot results
├── model/             ← checkpoints written here
├── logs/              ← TensorBoard + eval JSON written here
└── archive/           ← original monolithic files
```

## Configuration Reference (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TOTAL_TIMESTEPS` | 100,000 | Training length |
| `N_ENVS` | 1 | Parallel Isaac Sim worlds (scale freely: 1 → 100 → 1000) |
| `ISAAC_SIM_HEADLESS` | False | `True` = no viewport (recommended for N_ENVS ≥ 100) |
| `CHECKPOINT_FREQ` | 10,000 | Steps between checkpoint saves |
| `MAX_EPISODE_STEPS` | 300 | Episode truncation length |
| `MODEL_DIR` | `model/` | Where `.zip` files are saved |
| `LOG_DIR` | `logs/` | TensorBoard & eval JSON |
| `PPO` | dict | All SB3 PPO hyperparameters |

## CLI Reference

### `train.py`

```
usage: python.sh train.py [options]

  --timesteps N      Override TOTAL_TIMESTEPS from config.py
  --n-envs N         Number of parallel environments (e.g. 1, 100, 1000)
  --headless         Hide the Omniverse viewport (recommended for n-envs >= 100)
  --no-headless      Show the Omniverse viewport (default)
```

### `evaluate.py`

```
usage: python.sh evaluate.py [options]

  --model PATH       Path to model zip without extension
  --episodes N       Number of evaluation episodes
  --save-results F   Write per-episode JSON to file F
  --headless / --no-headless
```

## Performance Expectations

| Hardware | Envs | 10k steps time |
|----------|------|----------------|
| CPU only | 1 | 5–10 min |
| RTX 4090 | 1 | ~1–2 min |
| RTX 4090 | 100 | ~5–15 sec |
| RTX 4090 | 1000 | ~1–5 sec (headless recommended) |

## Troubleshooting

### Isaac Sim not found
```bash
source ~/isaacsim/setup_python_env.sh
```

### `omni.isaac` / `isaacsim` import error
Always launch scripts with Isaac Sim's own Python:
```bash
~/isaacsim/python.sh train.py
```

### CUDA out of memory
Reduce `N_ENVS` or `batch_size` in `config.py`.

### GPU not detected
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Nucleus / asset server unreachable
Isaac Sim fetches the Franka USD from Nucleus. Ensure you have an active
Omniverse account and that the service is reachable:
```bash
ping nucleus.omniverse.nvidia.com
```

## Additional Resources

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/app_isaacsim/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/omniverse/)
