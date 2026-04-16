#!/usr/bin/env python3
"""
Evaluation script for trained Franka Pick-and-Place models.

Must be launched via Isaac Sim's Python runtime:

    ~/isaacsim/python.sh evaluate.py
    ~/isaacsim/python.sh evaluate.py --model model/pick_place_ppo --episodes 10

Results (per-episode reward, touch/lift success) are printed to stdout
and optionally written to a JSON file for later analysis in analysis.ipynb.
"""

import argparse
import json
import os
import time

# ── 1. Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate a trained pick-and-place model")
parser.add_argument("--model",    type=str, default=None, help="Path to model (no .zip extension)")
parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--no-headless", dest="headless", action="store_false")
parser.add_argument("--save-results", type=str, default=None,
                    help="Optional path to write JSON results (e.g. logs/eval_results.json)")
args = parser.parse_args()

# ── 2. Boot Isaac Sim ─────────────────────────────────────────────────────────
try:
    from isaacsim import SimulationApp
except ModuleNotFoundError:
    from omni.isaac.kit import SimulationApp  # type: ignore

import config

headless = args.headless
simulation_app = SimulationApp({"headless": headless})
print(f"✓ SimulationApp started (headless={headless})")

# ── 3. Remaining imports (safe after SimulationApp) ───────────────────────────
import numpy as np
from stable_baselines3 import PPO

from env import PickPlaceEnv

# ── 4. Resolve runtime config ─────────────────────────────────────────────────
model_path   = args.model    or config.EVAL_MODEL_PATH
num_episodes = args.episodes or config.EVAL_EPISODES

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("=" * 70)
print("Evaluation: Franka Pick-and-Place")
print("=" * 70)
print(f"  Model    : {model_path}.zip")
print(f"  Episodes : {num_episodes}")
print()

model = PPO.load(model_path, device="cpu")
results = []

env = PickPlaceEnv(max_episode_steps=config.MAX_EPISODE_STEPS)

for ep in range(num_episodes):
    obs, _ = env.reset()

    ep_reward = 0.0
    touched   = False
    lifted    = False
    t0        = time.time()

    while env.step_count < env.max_episode_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

        ee_pos    = obs[14:17]
        block_pos = obs[17:20]
        dist      = float(np.linalg.norm(ee_pos - block_pos))

        if dist < 0.12 and obs[23] > 0.01:
            touched = True
        if block_pos[2] > 0.10:
            lifted = True

        if terminated or truncated:
            break

    ep_time = time.time() - t0

    result = {
        "episode":       ep + 1,
        "reward":        round(ep_reward, 3),
        "touched":       touched,
        "lifted":        lifted,
        "steps":         env.step_count,
        "elapsed_s":     round(ep_time, 2),
    }
    results.append(result)

    touch_icon = "✓" if touched else "✗"
    lift_icon  = "✓" if lifted  else "✗"
    print(
        f"  Ep {ep+1:>2}/{num_episodes}  "
        f"reward={ep_reward:+8.2f}  "
        f"touch={touch_icon}  lift={lift_icon}  "
        f"steps={env.step_count}"
    )

env.close()

# ── 6. Summary ────────────────────────────────────────────────────────────────
n_touch = sum(r["touched"] for r in results)
n_lift  = sum(r["lifted"]  for r in results)
avg_rew = sum(r["reward"]  for r in results) / num_episodes

print()
print("=" * 70)
print(f"  Touch success : {n_touch}/{num_episodes}  ({100*n_touch/num_episodes:.0f}%)")
print(f"  Lift  success : {n_lift}/{num_episodes}  ({100*n_lift/num_episodes:.0f}%)")
print(f"  Avg reward    : {avg_rew:+.2f}")
print("=" * 70)

# ── 7. Optionally persist results ─────────────────────────────────────────────
save_path = args.save_results
if save_path:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"model": model_path, "results": results}, f, indent=2)
    print(f"\n✓ Results saved to: {save_path}")

simulation_app.close()
