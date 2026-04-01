import numpy as np
import torch

# joint limit written in USD (degree)
SO101_FOLLOWER_USD_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# motor limit written in real device (normalized to related range)
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


def convert_leisaac_action_to_lerobot(action):
    """Sim radians → LeRobot motor-space degrees."""
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    processed_action = np.zeros_like(action)
    action_deg = action / np.pi * 180.0

    for idx, joint_name in enumerate(SO101_FOLLOWER_USD_JOINT_LIMITS):
        jl = SO101_FOLLOWER_USD_JOINT_LIMITS[joint_name]
        ml = SO101_FOLLOWER_MOTOR_LIMITS[joint_name]
        joint_range = jl[1] - jl[0]
        motor_range = ml[1] - ml[0]
        processed_action[:, idx] = (action_deg[:, idx] - jl[0]) / joint_range * motor_range + ml[0]

    return processed_action


def convert_lerobot_action_to_leisaac(action):
    """LeRobot motor-space degrees → sim radians."""
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    processed_action = np.zeros_like(action)

    for idx, joint_name in enumerate(SO101_FOLLOWER_USD_JOINT_LIMITS):
        ml = SO101_FOLLOWER_MOTOR_LIMITS[joint_name]
        jl = SO101_FOLLOWER_USD_JOINT_LIMITS[joint_name]
        motor_range = ml[1] - ml[0]
        joint_range = jl[1] - jl[0]
        processed_deg = (action[:, idx] - ml[0]) / motor_range * joint_range + jl[0]
        processed_action[:, idx] = processed_deg / 180.0 * np.pi

    return processed_action