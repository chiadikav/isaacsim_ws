"""
Franka Pick-and-Place Gymnasium Environment (Isaac Sim 4.x backend).

Import this module only after SimulationApp has been created and the
Isaac Sim runtime is fully initialised.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Isaac Sim 4.x API (isaacsim.core.* replaces omni.isaac.core.*)
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path


class PickPlaceEnv(gym.Env):
    """
    Isaac Sim 4.x environment for Franka Panda pick-and-place.

    Observation (31D):
      [0:7]   - Arm joint positions  (7 DOF)
      [7:14]  - Arm joint velocities (7 DOF)
      [14:17] - End-effector (panda_hand) XYZ position
      [17:20] - Block XYZ position
      [20:23] - Block XYZ velocity
      [23]    - Gripper width (sum of finger joint positions)
      [24:31] - Task milestone flags / padding

    Action (8D):
      [0:7]   - Arm joint position deltas (normalised -1 to 1)
      [7]     - Gripper command: -1 = open, +1 = close
    """

    # Joint position limits (Franka Panda)
    JOINT_LIMITS = 2.8973
    # Max gripper finger opening (metres)
    GRIPPER_MAX = 0.04
    # Settling steps after reset
    SETTLE_STEPS = 100
    # Franka USD asset path (relative to Nucleus assets root)
    FRANKA_USD = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    def __init__(self, max_episode_steps: int = 300, env_index: int = 0, render: bool = False):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.max_action_step = 0.05
        self._idx = env_index       # unique per env instance
        self._render = render       # True when viewport is open

        self.world = World(stage_units_in_meters=1.0, backend="numpy")

        self._load_scene()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    # Spacing between adjacent environments (metres)
    ENV_SPACING = 2.0

    def _load_scene(self):
        """Load Franka robot and target block into the world."""
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError(
                "Cannot reach Isaac Sim Nucleus asset server. "
                "Verify Isaac Sim installation and that Nucleus is reachable."
            )

        # Each env instance gets its own prim path so names never collide
        franka_prim = f"/World/Franka_{self._idx}"
        block_prim  = f"/World/target_block_{self._idx}"
        franka_name = f"franka_{self._idx}"
        block_name  = f"target_block_{self._idx}"

        # Offset each env along the Y axis so they don't overlap
        self._origin = np.array([0.0, self._idx * self.ENV_SPACING, 0.0])

        # Add a shared ground plane once (env 0 only); camera is set later
        # after all envs are loaded so we know the total span
        if self._idx == 0:
            GroundPlane(
                prim_path="/World/GroundPlane",
                z_position=0,
                color=np.array([0.12, 0.12, 0.15]),  # dark charcoal – contrasts with robot
            )

        # Step 1: add USD reference onto the stage
        franka_usd = assets_root + self.FRANKA_USD
        add_reference_to_stage(usd_path=franka_usd, prim_path=franka_prim)
        print(f"✓ Env {self._idx}: Franka USD staged")

        # Step 2: wrap the staged prim in a SingleArticulation for joint control
        self.franka = self.world.scene.add(
            SingleArticulation(
                prim_path=franka_prim,
                name=franka_name,
                position=self._origin,
            )
        )
        print(f"✓ Env {self._idx}: Franka articulation registered")

        # Step 3: EE prim path (unique per env)
        self._hand_prim_path = f"{franka_prim}/panda_hand"

        # Step 4: add target block (offset to this env's origin)
        self.target_block = self.world.scene.add(
            DynamicCuboid(
                prim_path=block_prim,
                name=block_name,
                position=self._origin + np.array([0.5, 0.0, 0.035]),
                scale=np.array([0.04, 0.04, 0.04]),
                color=np.array([1.0, 0.0, 0.0]),
                mass=0.1,
            )
        )
        print(f"✓ Env {self._idx}: Target block added")

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.world.reset()

        # Neutral arm pose (joints 0-6) + open gripper (joints 7-8)
        neutral_pose = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.5, 0.785,
                                 self.GRIPPER_MAX, self.GRIPPER_MAX])
        self.franka.set_joint_positions(neutral_pose)

        # Reset block to initial pose (relative to this env's origin)
        self.target_block.set_world_pose(position=self._origin + np.array([0.5, 0.0, 0.035]))
        self.target_block.set_linear_velocity(np.zeros(3))
        self.target_block.set_angular_velocity(np.zeros(3))

        # Let physics settle; only env 0 needs to drive the render flush
        _do_render = self._render and self._idx == 0
        for _ in range(self.SETTLE_STEPS):
            self.world.step(render=_do_render)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # --- Arm + gripper control ---
        current_q = self.franka.get_joint_positions()
        target_q = current_q[:7] + action[0:7] * self.max_action_step
        target_q = np.clip(target_q, -self.JOINT_LIMITS, self.JOINT_LIMITS)
        target_q[5] = -target_q[3]  # keep wrist parallel to gripper

        # Map gripper action [-1, 1] -> [GRIPPER_MAX, 0]
        finger_width = (1.0 - action[7]) / 2.0 * self.GRIPPER_MAX
        full_cmd = np.append(target_q, [finger_width, finger_width])
        self.franka.apply_action(ArticulationAction(joint_positions=full_cmd))

        self.world.step(render=self._render and self._idx == 0)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self.step_count >= self.max_episode_steps

        return obs, float(reward), terminated, False, {}

    def close(self):
        try:
            self.world.clear()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_ee_pos(self) -> np.ndarray:
        """Return world-frame XYZ position of panda_hand."""
        hand_prim = get_prim_at_path(self._hand_prim_path)
        pos, _ = get_world_pose_from_relative(
            hand_prim, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        )
        return np.array(pos, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(31, dtype=np.float32)

        all_q = self.franka.get_joint_positions()
        all_v = self.franka.get_joint_velocities()
        obs[0:7]   = all_q[:7]   # arm joints
        obs[7:14]  = all_v[:7]   # arm velocities
        obs[14:17] = self._get_ee_pos()
        obs[23]    = float(np.sum(all_q[7:9]))  # gripper width

        block_pos, _ = self.target_block.get_world_pose()
        block_vel    = self.target_block.get_linear_velocity()
        obs[17:20] = block_pos
        obs[20:23] = block_vel

        return obs

    def _compute_reward(self, obs: np.ndarray) -> float:
        reward = -0.001  # time penalty

        ee_pos    = obs[14:17]
        block_pos = obs[17:20]

        horiz_dist  = float(np.linalg.norm(ee_pos[:2] - block_pos[:2]))
        vert_dist   = float(ee_pos[2] - block_pos[2])
        ee_to_block = float(np.linalg.norm(ee_pos - block_pos))

        if horiz_dist < 0.15:
            reward += 1.0
        if horiz_dist < 0.08 and abs(vert_dist - 0.05) < 0.03:
            reward += 2.0
        if ee_to_block < 0.08:
            reward += 5.0
        if block_pos[2] > 0.10:  # lifted off table
            reward += 10.0

        return reward
