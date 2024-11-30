import gymnasium as gym
from omni.isaac.lab.app import AppLauncher
import numpy as np
# Initialize Isaac Sim
app_launcher = AppLauncher(headless=False)  # Set headless=True if you don't need visualization
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import omni.isaac.lab_tasks  

env_cfg = parse_env_cfg(
    "Swing-v0",
    device="cpu",  # or "cpu" if no GPU
    num_envs=2  # or however many environments you want
)

# Create the environment using gym
env = gym.make("Swing-v0", cfg=env_cfg)

# Reset the environment
obs, info = env.reset()

initial_action = torch.zeros((2, 21))  # Changed from (1, 21) to (100, 21)
# These values create a stable standing pose
# initial_action[:] = torch.tensor([
#     0.0,  # hip_x
#     0.0,  # hip_y # setting this negative makes robot lean forward at waist
#     0.0,  # hip_z # idk
#     -1.0,  # chest_x
#     0.0,  # chest_y
#     0.0,  # chest_z
#     0.0,  # neck
#     0.0,  # right_hip_x
#     0.0,  # right_hip_y (slight bend)
#     0.0, # right_knee (bent to stabilize
#     0.0,  # right_ankle
#     0.0,  # left_hip_x
#     0.0,  # left_hip_y (slight bend)
#     0.0, # left_knee (bent to stabilize)
#     0.0,  # left_ankle
#     0.0,  # right_shoulder_x
#     0.0,  # right_shoulder_y
#     0.0,  # right_elbow
#     0.0,  # left_shoulder_x
#     0.0,  # left_shoulder_y
#     0.0,  # left_elbow
# ])

# initial_action[:] = torch.tensor([
#     0,  # lower_waist
#     0.0,  # lower_waist
#     0.0,  # right_upper_arm
#     0.0, # right_upper_arm
#     0.0,  # left_upper_arm
#     0.0,  # left_upper_arm
#     0.0,  # pelvis
#     0.0,  # right_lower_arm
#     0.0,  # left_lower_arm
#     0.0,  # right_thigh: x
#     0.0,  # right_thigh: y
#     0.0,  # right_thigh: z
#     0.0,  # left_thigh: x
#     0.0,  # left_thigh: y
#     0.0,  # left_thigh: z
#     0.0,  # right_knee
#     0.0,  # left_knee
#     0.0,  # right_foot
#     0.0,  # right_foot
#     0.0,  # left_foot
#     0.0,
# ])

# initial_action[:] = torch.tensor([
#     0.0,  # lower_waist
#     0.0,  # lower_waist
#     -0.30267698 + 0.385398,  # right_upper_arm adduction (neg is up)
#     0.46815178 -0.615472907, # right_upper_arm flexion (pos is up)
#     0.3498035 -0.785398, #-2.35619,  # left_upper_arm adduction (pos is up)
#     0.519223 -0.815472907,#-2.526119746,  # left_upper_arm flexion (pos is up)
#     0.0,  # pelvis
#     -2.5708,  # right_lower_arm
#     -2.5708,  # left_lower_arm
#     0.0,  # right_thigh: x
#     0.0,  # right_thigh: y
#     0.0,  # right_thigh: z
#     0.0,  # left_thigh: x
#     0.0,  # left_thigh: y-
#     0.0,  # left_thigh: z
#     0.0,  # right_knee
#     0.0,  # left_knee
#     0.0,  # right_foot
#     0.0,  # right_foot
#     0.0,  # left_foot
#     0.0,  # left_foot
# ])



initial_action[:] = torch.tensor([
    0,  # lower_waist
    0,  # lower_waist
    0.385398,  # right_upper_arm adduction (neg is up)
    -0.615472907, # right_upper_arm flexion (pos is up)
    -0.785398, #-2.35619,  # left_upper_arm adduction (pos is up)
    -0.815472907,#-2.526119746,  # left_upper_arm flexion (pos is up)
    0.0,  # pelvis
    -2,  # right_lower_arm
    -2,  # left_lower_arm
    0.0,  # right_thigh: x
    0.0,  # right_thigh: y
    0.0,  # right_thigh: z
    0.0,  # left_thigh: x
    0.0,  # left_thigh: y-
    0.0,  # left_thigh: z
    0.0,  # right_knee
    0.0,  # left_knee
    0.0,  # right_foot
    0.0,  # right_foot
    0.0,  # left_foot
    0.0,  # left_foot
])

offset = torch.tensor([
    0.8143025,  # lower_waist
    -0.89417243,  # lower_waist
    0.4-0.30267698,  # right_upper_arm adduction (neg is up)
    0.7 + 0.46815178, # right_upper_arm flexion (pos is up)
    0.3498035,
    0.519223, #-2.35619,  # left_upper_arm adduction (pos is up)
    0.0, # pelvis
    0.85204905, # right_lower_arm
    0.8723765, # left_lower_arm
    -0.14598082,  # right_thigh: x
    -0.77993596,  # right_thigh: y
    0.0,  # right_thigh: z
    0.14598082,  # left_thigh: x
    -0.836486,  # left_thigh: y-
    0.0,  # left_thigh: z
    -0.11140303,  # right_knee
    -0.11913639,  # left_knee
    0.0,  # right_foot
    0.0,  # right_foot
    0.0,  # left_foot
    0.0,  # left_foot
])

initial_action = initial_action *0


# [ 0.8143025  -0.89417243  0.30267698  0.46815178 -0.3498035   0.519223
#  -0.85204905 -0.8723765  -0.77993596 -0.04598082  0.0260668   0.836486
#   0.08649698  0.08993983  0.25654003  0.11140303  0.11913639 -0.22328617
#   0.34219608 -0.16721205  0.3553766 ]

# In your simulation loop, use PD control to maintain the pose
kp = 5  # Proportional gain
kd = 0   # Derivative gain

actions = initial_action
target_pose = None
# Run simulation loop
while simulation_app.is_running():
    # Step the environment
   
    obs, reward, terminated, truncated, info = env.step(actions)
    print("??????????????????????????")
    print(obs['policy'])
    print("??????????????????????????")
    if target_pose is None:
        target_pose = obs['policy']['joint_pos_norm']
    # Sample random actions (in a real scenario, you'd use your policy here)
    joint_pos = obs['policy']['joint_pos_norm']
    joint_vel = obs['policy']['joint_vel_rel']

    
    
    # Compute PD control
    position_error = target_pose - joint_pos
    actions = kp * position_error - kd * joint_vel
    actions = initial_action
    # Clip actions to valid range
    # actions = torch.clamp(actions, -1.0, 1.0)
    # print(actions)
    
    print("YOOOOOOOOOOOOOOOOO")
    # Reset if any environment is done
    if terminated.any() or truncated.any():
        obs, info = env.reset()
        # Optionally reset target_pose here if needed
        # target_pose = None

# Cleanup
env.close()
simulation_app.close()