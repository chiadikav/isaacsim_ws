# MuJoCo vs Isaac Sim – Comparison

## Physics Engine

| Aspect | MuJoCo | Isaac Sim |
|--------|--------|-----------|
| Engine type | Spring-damper (MuJoCo ODE) | Constraint-based (Omniverse Physics / TGS) |
| Primary compute | CPU | GPU |
| Contact model | Spring-damper | Physics-based constraints |
| Model format | XML (MJCF) | USD (Universal Scene Description) |
| Installation size | ~10 MB | ~20 GB |
| GPU support | Inference only | Full simulation |

## Performance

| Metric | MuJoCo | Isaac Sim |
|--------|--------|-----------|
| 100k steps, CPU | ~10–15 min | N/A |
| 100k steps, GPU | N/A | ~2–3 min |
| Practical `n_envs` | 4–8 (CPU-bound) | 16–32+ (GPU-bound) |
| Real-time factor (32 envs) | ~1× | ~20–50× |

## API Comparison

### World / Model Setup

```python
# MuJoCo
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")
data  = mujoco.MjData(model)

# Isaac Sim
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
world  = World(backend="torch", device="cuda")
franka = world.scene.add(Robot(prim_path="/World/Franka", usd_path="..."))
```

### Simulation Step

```python
# MuJoCo
mujoco.mj_step(model, data)

# Isaac Sim
world.step(render=False)
```

### Joint Control

```python
# MuJoCo
data.ctrl[:7] = target_positions
mujoco.mj_step(model, data)

# Isaac Sim
from omni.isaac.core.utils.types import ArticulationAction
franka.apply_action(ArticulationAction(joint_positions=target_positions))
world.step(render=False)
```

### Observations

```python
# MuJoCo
arm_pos = data.qpos[0:7]
arm_vel = data.qvel[0:7]
ee_pos  = data.body("hand").xpos

# Isaac Sim
arm_pos = franka.get_joint_positions()[:7]
arm_vel = franka.get_joint_velocities()[:7]
ee_pos  = franka.get_link_positions(link_indices=np.array([10]))[0]
```

## Observation & Action Spaces

Both versions use **identical** spaces so a model trained in one engine can be
drop-in evaluated in the other:

| Space | Shape | Description |
|-------|-------|-------------|
| Observation | 31D | arm pos/vel, EE pos, block pos/vel, gripper |
| Action | 8D | 7 joint deltas + gripper command |

```python
# Cross-engine evaluation example
mujoco_model = PPO.load("mujoco_model")
env_isaac = PickPlaceEnv()
obs, _ = env_isaac.reset()
action, _ = mujoco_model.predict(obs)   # compatible!
```

## Reward Structure (identical in both)

```
reward = -0.001                           # time penalty
+ 1.0   if horizontal dist < 0.15 m      # approaching
+ 2.0   if in pre-grasp position
+ 5.0   if 3-D dist < 0.08 m             # very close
+ 10.0  if block lifted > 0.10 m
```

## Feature Parity

| Feature | MuJoCo | Isaac Sim |
|---------|--------|-----------|
| Joint position control | ✓ | ✓ |
| Gripper control | ✓ | ✓ |
| Contact detection | ✓ | ✓ |
| Real-time visualisation | ✗ | ✓ (Omniverse GUI) |
| GPU-parallel training | ✗ | ✓ |
| Stable Baselines3 compat | ✓ | ✓ |
| Gymnasium API | ✓ | ✓ |

## Which Should I Use?

**Use MuJoCo if:**
- Running on CPU-only hardware
- Need a lightweight, fast setup
- Want deterministic, stable physics
- Rapid prototyping / CI environments

**Use Isaac Sim if:**
- NVIDIA GPU is available
- Need GPU-accelerated or large-scale parallel training
- Targeting sim-to-real transfer
- Require photorealistic rendering or sensor simulation
- Integrating with the broader Omniverse ecosystem

## Known Physics Differences

| Behaviour | MuJoCo | Isaac Sim | Mitigation |
|-----------|--------|-----------|------------|
| Contact stiffness | Spring-damper | Constraint solver | Re-tune reward thresholds |
| Gripper friction | Configurable via XML | USD physics material | Adjust USD material props |
| Stability | Excellent with implicit integrator | Excellent with TGS solver | Use TGS solver in Isaac Sim |
| Determinism | High (fixed seed) | High (fixed seed) | Set numpy/torch seeds |

## Migration Checklist (MuJoCo → Isaac Sim)

- [ ] Replace `mujoco.MjModel` / `MjData` with `World` + `Robot`
- [ ] Replace `mj_step()` with `world.step()`
- [ ] Replace `data.ctrl` with `apply_action(ArticulationAction(...))`
- [ ] Replace `data.qpos/qvel` with `get_joint_positions/velocities()`
- [ ] Convert model from XML to USD (use Isaac Sim URDF importer)
- [ ] Adjust physics parameters (timestep default differs)
- [ ] Launch scripts via `python.sh` instead of system Python
