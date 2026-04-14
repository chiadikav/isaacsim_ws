# Quick Start Guide

Get up and running with Franka Pick-and-Place in under 5 minutes.

## TL;DR

```bash
ISAAC_PYTHON=~/isaacsim/python.sh

$ISAAC_PYTHON train.py                                             # train (100 k steps, UI open)
$ISAAC_PYTHON train.py --timesteps 500000 --n-envs 100             # 100 parallel envs
$ISAAC_PYTHON train.py --timesteps 1000000 --n-envs 1000 --headless  # large scale, no UI
$ISAAC_PYTHON evaluate.py --episodes 10 --save-results logs/eval_results.json
# then open analysis.ipynb to plot results
```

## Prerequisites Checklist

- [ ] NVIDIA GPU installed
- [ ] NVIDIA driver installed (`nvidia-smi` works)
- [ ] Isaac Sim installed at `~/isaacsim`
- [ ] Python dependencies installed (`gymnasium`, `stable-baselines3`, `torch`)

## One-Time Setup

```bash
# Source the Isaac Sim environment
source ~/isaacsim/setup_python_env.sh

# Install RL dependencies inside that environment
pip install gymnasium stable-baselines3 torch
```

For a full walkthrough see [SETUP.md](SETUP.md).

## Training

### Quick test (~1 min, verifies everything works)

```bash
$ISAAC_PYTHON train.py --timesteps 10000
```

### Standard training (~3 min on RTX 4090)

```bash
$ISAAC_PYTHON train.py --timesteps 100000
```

### Full training with parallel environments

```bash
# 100 envs – UI stays open (default)
$ISAAC_PYTHON train.py --timesteps 500000 --n-envs 100

# 1000 envs – suppress UI to save VRAM
$ISAAC_PYTHON train.py --timesteps 1000000 --n-envs 1000 --headless
```

The Omniverse viewport opens by default. Use `--headless` when running large
environment counts (≥ 100) to reduce GPU memory overhead.

Checkpoints are saved automatically to `model/` every 10,000 steps.
Killing training with **Ctrl-C** saves the current model before exiting.

## Evaluation

```bash
# Quick check (5 episodes, default model)
$ISAAC_PYTHON evaluate.py

# Thorough evaluation, save results for plotting
$ISAAC_PYTHON evaluate.py --episodes 20 --save-results logs/eval_results.json

# Evaluate a specific checkpoint
$ISAAC_PYTHON evaluate.py --model model/pick_place_ppo_50000_steps
```

## Analysing Results

Open `analysis.ipynb` in VS Code with any standard Python 3 kernel (no Isaac Sim
needed). It reads TensorBoard logs from `logs/` and evaluation JSON from
`logs/eval_results.json` and plots reward curves and success rates.

## Tuning Hyperparameters

Edit `config.py` — that is the **only** file you need to change for routine
adjustments:

```python
# config.py
TOTAL_TIMESTEPS = 500_000
N_ENVS          = 100      # scale freely: 1, 100, 1000, ...
ISAAC_SIM_HEADLESS = False # True = no UI (recommended for N_ENVS >= 100)
CHECKPOINT_FREQ = 10_000

PPO = dict(
    learning_rate = 3e-4,
    n_steps       = 1024,
    batch_size    = 64,
    ...
)
```

## Troubleshooting

### `ModuleNotFoundError: omni.isaac`
You must use Isaac Sim's own Python runtime:
```bash
~/isaacsim/python.sh train.py
```

### `CUDA out of memory`
Reduce parallel environments or batch size in `config.py`:
```python
N_ENVS = 1
PPO = dict(batch_size=32, ...)
```

### GPU not detected
```bash
nvidia-smi          # verify driver
python -c "import torch; print(torch.cuda.is_available())"
```

### Training is very slow
Confirm CUDA is available (above) and that you launched via `python.sh`, not a
system Python.

## Command Reference

```bash
# Train (UI open by default; add --headless for large env counts)
$ISAAC_PYTHON train.py [--timesteps N] [--n-envs N] [--headless | --no-headless]

# Evaluate
$ISAAC_PYTHON evaluate.py [--model PATH] [--episodes N] [--save-results PATH]

# Monitor training live
tensorboard --logdir logs/

# Watch GPU usage
watch -n 1 nvidia-smi
```

## Performance Targets

| Metric | Target | Approx. timesteps |
|--------|--------|-------------------|
| Touch success | > 70% | 50k |
| Lift success | > 40% | 100k |
| Consistent lift | > 60% | 200k+ |
