#!/usr/bin/env python3
"""
Training entry point for Franka Pick-and-Place RL.

Must be launched via Isaac Sim's Python runtime so that the Omniverse
runtime and USD/PhysX libraries are available:

    ~/isaacsim/python.sh train.py
    ~/isaacsim/python.sh train.py --timesteps 500000 --n-envs 100
    ~/isaacsim/python.sh train.py --timesteps 1000000 --n-envs 1000 --headless

By default the Omniverse viewport opens (non-headless). Pass --headless to
suppress it, which is recommended for very large environment counts.

SimulationApp MUST be created before any omni.* imports – this script
handles that correctly.
"""

import argparse
import os
import sys
import time

# ── 1. Parse args before touching Isaac Sim ──────────────────────────────────
parser = argparse.ArgumentParser(description="Train Franka pick-and-place with PPO")
parser.add_argument("--timesteps", type=int,   default=None,  help="Override TOTAL_TIMESTEPS from config")
parser.add_argument("--n-envs",    type=int,   default=None,  help="Number of parallel environments (e.g. 1, 100, 1000)")
parser.add_argument("--headless",  action="store_true",       help="Hide the Omniverse viewport (recommended for n-envs >= 100)")
parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show the Omniverse viewport (default)")
parser.set_defaults(headless=None)
args = parser.parse_args()

# ── 2. Boot Isaac Sim (SimulationApp must come first) ────────────────────────
try:
    from isaacsim import SimulationApp
except ModuleNotFoundError:
    from omni.isaac.kit import SimulationApp  # type: ignore  (Isaac Sim < 4.0)

import config  # noqa: E402 – config has no omni deps, safe to import early

headless = args.headless if args.headless is not None else config.ISAAC_SIM_HEADLESS
simulation_app = SimulationApp({
    "headless": headless,
    "width":    config.VIEWPORT_WIDTH,
    "height":   config.VIEWPORT_HEIGHT,
})
print(f"✓ SimulationApp started (headless={headless}, "
      f"{config.VIEWPORT_WIDTH}×{config.VIEWPORT_HEIGHT})")

# Enable camera light mode by default so the scene is always visible
# (equivalent to Viewport → Lighting → Camera Light)
if not headless:
    import carb
    import omni.kit.app
    from omni.kit.viewport.utility import get_active_viewport

    carb.settings.get_settings().set("/rtx/useViewLightingMode", True)

    # Set the viewport texture resolution so pixels are 1:1 with the window.
    # Two update() calls let the viewport finish initialising before we resize.
    omni.kit.app.get_app().update()
    omni.kit.app.get_app().update()
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.set_texture_resolution([config.VIEWPORT_WIDTH, config.VIEWPORT_HEIGHT])

# ── 3. Now safe to import omni.* and project modules ─────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env import PickPlaceEnv  # noqa: E402

# ── 4. Resolve runtime config ─────────────────────────────────────────────────
total_timesteps = args.timesteps if args.timesteps is not None else config.TOTAL_TIMESTEPS
n_envs          = args.n_envs    if args.n_envs    is not None else config.N_ENVS
# PPO with MlpPolicy is faster on CPU; leave GPU free for Isaac Sim physics
device = "cpu"

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR,   exist_ok=True)

model_save_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("Training: Franka Pick-and-Place (Isaac Sim + PPO)")
print("=" * 70)
print(f"  Timesteps : {total_timesteps:,}")
print(f"  Envs      : {n_envs}")
print(f"  PPO device: CPU  (Isaac Sim physics on GPU)")
print(f"  Model dir : {config.MODEL_DIR}")
print()

env = DummyVecEnv([
    (lambda i: lambda: PickPlaceEnv(
        max_episode_steps=config.MAX_EPISODE_STEPS,
        env_index=i,
        render=not headless,
    ))(idx)
    for idx in range(n_envs)
])

# Point the viewport camera at the centre of the robot row
if not headless:
    from isaacsim.core.utils.viewports import set_camera_view
    spacing = PickPlaceEnv.ENV_SPACING
    centre_y = (n_envs - 1) * spacing / 2.0
    set_camera_view(
        eye=[(n_envs * spacing * 0.6), centre_y, n_envs * spacing * 0.5],
        target=[0.0, centre_y, 0.5],
        camera_prim_path="/OmniverseKit_Persp",
    )

checkpoint_cb = CheckpointCallback(
    save_freq=max(config.CHECKPOINT_FREQ // n_envs, 1),
    save_path=config.MODEL_DIR,
    name_prefix=config.MODEL_NAME,
    verbose=1,
)

model = PPO(
    "MlpPolicy",
    env,
    **config.PPO,
    verbose=1,
    tensorboard_log=config.LOG_DIR,
    device=device,
)

start = time.time()
try:
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
except KeyboardInterrupt:
    print("\n⚠ Training interrupted – saving current model.")

elapsed = time.time() - start
model.save(model_save_path)

print("\n" + "=" * 70)
print(f"✓ Training complete in {elapsed / 60:.1f} min")
print(f"✓ Final model saved to: {model_save_path}.zip")
print("=" * 70)

env.close()
simulation_app.close()
