#!/usr/bin/env python3
"""
Franka Pick-and-Place RL Training Script - Isaac Sim Edition

This script trains a Franka Panda robot to pick up objects using PPO 
reinforcement learning in NVIDIA Isaac Sim.

Usage:
    python pick_and_place_train.py --timesteps 100000 --headless
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import torch
import argparse
import os

# Isaac Sim imports
try:
    from omni.isaac.kit import SimulationApp
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.types import ArticulationAction
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    print("⚠ Warning: Isaac Sim not available. Please install NVIDIA Isaac Sim.")
    ISAAC_SIM_AVAILABLE = False


class PickPlaceEnvIsaacSim(gym.Env):
    """
    Isaac Sim environment for Franka Panda pick and place task.
    
    Observation Space (31D):
    - [0:7]: Arm joint positions
    - [7:14]: Arm joint velocities
    - [14:17]: End-effector position
    - [17:20]: Block position
    - [20:23]: Block velocity
    - [23]: Gripper width
    - [24-28]: Task milestone flags
    - [29:31]: Padding
    
    Action Space (8D):
    - [0:7]: Joint velocity commands (normalized)
    - [7]: Gripper command (-1=open, +1=close)
    """
    
    def __init__(self, headless=False):
        super().__init__()
        
        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError("Isaac Sim is required but not available")
        
        # Initialize Isaac Sim world
        self.world = World(backend="torch", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Load robot and environment
        self._load_scene()
        
        # Configuration
        self.max_episode_steps = 300
        self.step_count = 0
        self.dt = 0.01
        
        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # Control parameters
        self.joint_forces = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0])
        self.max_action_step = 0.05
    
    def _load_scene(self):
        """Load Franka and scene objects."""
        try:
            # Load Franka robot
            franka_usd = "omniverse://nucleus.omniverse.nvidia.com/Isaac/Robots/Franka/franka.usd"
            self.franka = self.world.scene.add(
                Robot(
                    prim_path="/World/Franka",
                    name="franka",
                    usd_path=franka_usd
                )
            )
            print("✓ Franka robot loaded")
        except Exception as e:
            print(f"✗ Failed to load Franka: {e}")
            self.franka = None
        
        # Add target block
        try:
            from omni.isaac.core.prims import XFormPrim
            self.target_block = self.world.scene.add(
                XFormPrim(
                    prim_path="/World/target_block",
                    name="target_block",
                    position=np.array([0.5, 0.0, 0.035])
                )
            )
            print("✓ Target block added")
        except Exception as e:
            print(f"✗ Failed to add target block: {e}")
            self.target_block = None
    
    def reset(self, seed=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.step_count = 0
        
        try:
            self.world.reset()
            
            if self.franka is not None:
                neutral_pose = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.5, 0.785])
                self.franka.set_joint_positions(neutral_pose[:7])
                self.franka.set_joint_positions(np.array([0.04, 0.04]), joint_indices=np.array([7, 8]))
            
            if self.target_block is not None:
                self.target_block.set_world_pose(position=np.array([0.5, 0.0, 0.035]))
            
            # Settle simulation
            for _ in range(100):
                self.world.step(render=False)
        
        except Exception as e:
            print(f"⚠ Reset error: {e}")
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get observation vector."""
        obs = np.zeros(31, dtype=np.float32)
        
        try:
            if self.franka is not None:
                arm_pos = self.franka.get_joint_positions()[:7]
                arm_vel = self.franka.get_joint_velocities()[:7]
                obs[0:7] = arm_pos
                obs[7:14] = arm_vel
                
                # End-effector position (hand link)
                ee_pos = self.franka.get_link_positions(link_indices=np.array([10]))[0]
                obs[14:17] = ee_pos
            
            if self.target_block is not None:
                block_pos, _ = self.target_block.get_world_pose()
                block_vel = self.target_block.get_linear_velocity()
                obs[17:20] = block_pos
                obs[20:23] = block_vel
        
        except Exception as e:
            print(f"⚠ Observation error: {e}")
        
        return obs
    
    def step(self, action):
        """Execute action step."""
        action = np.clip(action, -1.0, 1.0)
        
        try:
            if self.franka is not None:
                current_q = self.franka.get_joint_positions()
                
                # Arm control
                arm_deltas = action[0:7] * self.max_action_step
                target_q = current_q[:7] + arm_deltas
                target_q = np.clip(target_q, -2.8973, 2.8973)
                target_q[5] = -target_q[3]  # Keep gripper parallel
                
                articulation_action = ArticulationAction(joint_positions=target_q)
                self.franka.apply_action(articulation_action)
                
                # Gripper control
                gripper_width = (1.0 - action[7]) / 2.0 * 0.04
                self.franka.set_joint_positions(
                    np.array([gripper_width, gripper_width]),
                    joint_indices=np.array([7, 8])
                )
            
            self.world.step(render=False)
        
        except Exception as e:
            print(f"⚠ Step error: {e}")
        
        self.step_count += 1
        obs = self._get_obs()
        
        # Calculate reward
        reward = -0.001
        
        try:
            if self.franka is not None and self.target_block is not None:
                ee_pos = self.franka.get_link_positions(link_indices=np.array([10]))[0]
                block_pos, _ = self.target_block.get_world_pose()
                
                horiz_dist = np.linalg.norm(ee_pos[0:2] - block_pos[0:2])
                vert_dist = ee_pos[2] - block_pos[2]
                ee_to_block = np.linalg.norm(ee_pos - block_pos)
                
                # Milestone rewards
                if horiz_dist < 0.15:
                    reward += 1.0
                if horiz_dist < 0.08 and abs(vert_dist - 0.05) < 0.03:
                    reward += 2.0
                if ee_to_block < 0.08:
                    reward += 5.0
        
        except Exception as e:
            print(f"⚠ Reward error: {e}")
        
        terminated = self.step_count >= self.max_episode_steps
        
        return obs, float(reward), terminated, False, {}
    
    def close(self):
        """Clean up."""
        try:
            if self.world is not None:
                self.world.clear()
        except Exception as e:
            print(f"⚠ Close error: {e}")


def train(total_timesteps=100000, n_envs=1, headless=True):
    """Train the agent."""
    print("=" * 70)
    print("Training: Franka Pick and Place (Isaac Sim)")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Device: {('CUDA' if torch.cuda.is_available() else 'CPU')}")
    print()
    
    def make_env():
        return PickPlaceEnvIsaacSim(headless=headless)
    
    try:
        env = make_vec_env(make_env, n_envs=n_envs)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            policy_kwargs={"net_arch": [256, 256]},
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps)
        elapsed = time.time() - start_time
        
        model.save("pick_block_ppo_isaac")
        
        print("\n" + "=" * 70)
        print(f"✓ Training complete in {elapsed/60:.1f} minutes")
        print(f"✓ Model saved as 'pick_block_ppo_isaac.zip'")
        print("=" * 70)
        
        env.close()
    
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


def evaluate(model_path="pick_block_ppo_isaac", num_episodes=3):
    """Evaluate the trained model."""
    print("=" * 70)
    print("Evaluating: Franka Pick and Place")
    print("=" * 70)
    
    try:
        model = PPO.load(model_path)
        
        touches = 0
        for ep in range(num_episodes):
            env = PickPlaceEnvIsaacSim(headless=True)
            obs, _ = env.reset()
            
            touched = False
            while env.step_count < env.max_episode_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                
                ee_pos = obs[14:17]
                block_pos = obs[17:20]
                distance = np.linalg.norm(ee_pos - block_pos)
                
                if distance < 0.12:
                    touched = True
                
                if terminated or truncated:
                    break
            
            if touched:
                touches += 1
            
            print(f"Episode {ep+1}: {'✓ Touch' if touched else '✗ Miss'}")
            env.close()
        
        print(f"\nSuccess rate: {touches}/{num_episodes} ({100*touches/num_episodes:.0f}%)")
    
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka Pick and Place RL Training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
    parser.add_argument("--eval", action="store_true", help="Evaluate model instead of training")
    parser.add_argument("--model", type=str, default="pick_block_ppo_isaac", help="Model path for evaluation")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(model_path=args.model, num_episodes=args.episodes)
    else:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs, headless=args.headless)
