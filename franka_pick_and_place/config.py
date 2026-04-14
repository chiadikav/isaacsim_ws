"""
Centralised configuration for Franka Pick-and-Place RL.

Edit values here rather than hunting through train.py / evaluate.py.
"""

from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Directory where trained model checkpoints are written.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Base name used for the final saved model (no extension).
MODEL_NAME = "pick_place_ppo"

# ---------------------------------------------------------------------------
# Isaac Sim
# ---------------------------------------------------------------------------

ISAAC_SIM_HEADLESS = False         # set True to suppress the Omniverse viewport

# Viewport window resolution (pixels). 1920×1080 gives a standard 16:9 ratio.
VIEWPORT_WIDTH  = 1920
VIEWPORT_HEIGHT = 1080

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

MAX_EPISODE_STEPS = 300            # timesteps per episode before truncation

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

PPO = dict(
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs={"net_arch": [256, 256]},
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 100_000
N_ENVS = 1                         # parallel Isaac Sim worlds – scale freely (e.g. 100, 1000)
CHECKPOINT_FREQ = 10_000           # save a checkpoint every N timesteps
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL_EPISODES = 5
EVAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
