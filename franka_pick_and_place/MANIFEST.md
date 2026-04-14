# Project Manifest

Complete inventory of every file in `franka_pick_and_place/`.

## Source Files

### `config.py`
Single source of truth for all hyperparameters and directory paths.
Edit this file first before changing anything else.

| Symbol | Default | Purpose |
|--------|---------|----------|
| `MODEL_DIR` | `model/` | Where checkpoints are saved |
| `MODEL_NAME` | `pick_place_ppo` | Base filename for saved models |
| `TOTAL_TIMESTEPS` | 100,000 | Training length |
| `N_ENVS` | 1 | Parallel Isaac Sim worlds |
| `CHECKPOINT_FREQ` | 10,000 | Steps between checkpoint saves |
| `MAX_EPISODE_STEPS` | 300 | Episode truncation |
| `LOG_DIR` | `logs/` | TensorBoard + eval JSON |
| `PPO` | dict | All SB3 PPO constructor kwargs |
| `EVAL_EPISODES` | 5 | Default evaluation episodes |
| `EVAL_MODEL_PATH` | `model/pick_place_ppo` | Default model for evaluation |

### `env.py`
Defines `PickPlaceEnv(gym.Env)` – the Gymnasium-compatible Isaac Sim environment.

Important: this module may only be imported **after** `SimulationApp` has been
created (i.e. after `train.py` or `evaluate.py` have booted Isaac Sim).

**Public API:**

| Method | Description |
|--------|-------------|
| `__init__(max_episode_steps)` | Create world, load Franka + block |
| `reset(seed, options)` | Reset world, return `(obs, info)` |
| `step(action)` | Apply action, return `(obs, reward, terminated, truncated, info)` |
| `close()` | Tear down Isaac Sim world |

### `train.py`
Training entry point. **Must be launched via Isaac Sim's `python.sh`.**

```bash
~/isaacsim/python.sh train.py [--timesteps N] [--n-envs N] [--headless | --no-headless]
```

- Boots `SimulationApp` before any `omni.*` imports
- Opens the Omniverse viewport by default (`--headless` suppresses it)
- `--n-envs` accepts any positive integer — use 100 or 1000 for large-scale GPU training
- Uses `CheckpointCallback` to save `model/pick_place_ppo_<N>_steps.zip` periodically
- Saves final model on completion **or** on `Ctrl-C`
- Writes TensorBoard logs to `logs/`

### `evaluate.py`
Evaluation script. **Must be launched via Isaac Sim's `python.sh`.**

```bash
~/isaacsim/python.sh evaluate.py [--model PATH] [--episodes N] [--save-results logs/eval.json]
```

- Boots `SimulationApp` before any `omni.*` imports
- Reports per-episode reward, touch success, lift success
- Optionally writes a JSON file for use by `analysis.ipynb`

## Notebook

### `analysis.ipynb`
Post-training visualisation. **No Isaac Sim required** — runs under any Python 3 kernel.

| Cell | Content |
|------|---------|
| 1 | Imports and path setup |
| 2 | Load TensorBoard CSV (Option A) |
| 3 | Load TFEvents directly via `tensorboard` library (Option B) |
| 4 | Plot reward curve with rolling mean |
| 5 | Load `logs/eval_results.json` |
| 6 | Bar charts: reward, touch/lift success, episode length |

## Documentation

| File | Audience | Purpose |
|------|----------|---------|
| `README.md` | Everyone | Project overview, structure, quick start |
| `QUICKSTART.md` | Returning users | Commands, tuning, troubleshooting |
| `SETUP.md` | New users / admins | Step-by-step installation, CLI reference |
| `COMPARISON.md` | ML engineers | MuJoCo vs Isaac Sim API and physics |
| `MANIFEST.md` | Maintainers | This file |

## Directories

| Directory | Created by | Contents |
|-----------|------------|----------|
| `model/` | `train.py` | `pick_place_ppo.zip`, `pick_place_ppo_<N>_steps.zip` |
| `logs/` | `train.py` / `evaluate.py` | TensorBoard event files, `eval_results.json` |
| `archive/` | Manual | `pick_and_place_train.py`, `pick_and_place_train.ipynb` (original monolith) |

## Generated Outputs (after training)

| File | Source | Description |
|------|--------|-------------|
| `model/pick_place_ppo.zip` | `train.py` | Final trained model |
| `model/pick_place_ppo_<N>_steps.zip` | `train.py` | Periodic checkpoints |
| `logs/PPO_*/events.out.*` | `train.py` | TensorBoard event files |
| `logs/eval_results.json` | `evaluate.py --save-results` | Per-episode evaluation data |

## Compatibility

| Aspect | Status |
|--------|--------|
| Gymnasium API | ✓ Standard `(obs, reward, terminated, truncated, info)` |
| Stable Baselines3 | ✓ Any `on-policy` or `off-policy` algorithm |
| Isaac Sim 4.x | ✓ Tested (`from isaacsim import SimulationApp`) |
| Isaac Sim < 4.0 | ✓ Fallback to `from omni.isaac.kit import SimulationApp` |
| Cross-engine eval | ✓ MuJoCo models evaluate in this env (identical spaces) |

## Maintenance Notes

- All hyperparameter changes → `config.py` only
- Environment logic changes → `env.py` only; `train.py` / `evaluate.py` need no edits
- Physics tuning → adjust `_compute_reward()` and `SETTLE_STEPS` in `env.py`
- To add a new metric to evaluation → extend `evaluate.py` result dict and add a cell to `analysis.ipynb`

---

**Created**: April 13, 2026  
**Isaac Sim version**: 4.0+  
**Status**: Production ready
