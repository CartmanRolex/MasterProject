import time
import torch
import numpy as np
from collections import deque
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import build_inference_frame

from robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

# Shared Evaluation Utilities
from eval_utils import (
    save_positions, 
    count_oranges_in_plate, 
    save_camera_snapshots, 
    EvaluationTracker
)

SINGLE_ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "MasterProject2026/ACT-pick-orange"
n_episodes = 100
instruction = "Grab orange and place into plate"
actions_per_chunk = 50  

dataset_features = {
    "observation.images.front": {"dtype": "video", "shape": (3, 480, 640), "names": ["front"]},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["wrist"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (6,), "names": ["action"]},
}

# ==========================================
# 2. Environment & Policy Initialization
# ==========================================
print("Loading LeIsaac Environment...")
envs_dict = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][0].envs[0].unwrapped

print(f"Loading trained ACT policy: {model_id}...")
policy = ACTPolicy.from_pretrained(model_id).to(device).eval()

device_override = {"device": str(device)}
preprocess, postprocess = make_pre_post_processors(
    policy.config, 
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": device_override},
    postprocessor_overrides={"device_processor": device_override},
)

# ==========================================
# 3. Evaluation Loop
# ==========================================
tracker = EvaluationTracker(n_episodes)
print(f"\n--- STARTING EVALUATION: {n_episodes} EPISODES ---")

try:
    for episode in range(n_episodes):
        obs, _ = env.reset()
        policy.reset() 
        
        done = False
        step_count = 0
        action_queue = deque()
        tracker.start_episode(episode)

        while not done:
            policy_obs = obs['policy']

            # Extract raw images every step to allow for snapshotting
            raw_front = policy_obs['front'][0].cpu().numpy()
            raw_wrist = policy_obs['wrist'][0].cpu().numpy()
            save_camera_snapshots(raw_front, raw_wrist, episode, step_count)

            infer_time_ms = None
            
            # === SERVER-CLIENT MOCK PIPELINE ===
            if len(action_queue) == 0:
                joint_pos_raw = policy_obs['joint_pos'].cpu().numpy()
                joint_pos_converted = convert_leisaac_action_to_lerobot(joint_pos_raw)

                raw_observations = {
                    "front": raw_front,
                    "wrist": raw_wrist,
                    "state": joint_pos_converted[0]
                }

                obs_frame = build_inference_frame(
                    observation=raw_observations, 
                    ds_features=dataset_features,
                    device=device, 
                    task=instruction
                )

                batch = preprocess(obs_frame)
                
                # Strip temporal dimension for ACT
                if "observation.state" in batch and batch["observation.state"].ndim == 3: 
                    batch["observation.state"] = batch["observation.state"][:, 0, :]
                for cam in ["observation.images.front", "observation.images.wrist"]:
                    if cam in batch and batch[cam].ndim == 5: 
                        batch[cam] = batch[cam][:, 0, :, :, :]
                
                # --- Inference ---
                t_infer_start = time.perf_counter()
                with torch.inference_mode():
                    action_tensor = policy.predict_action_chunk(batch)
                t_infer_end = time.perf_counter()
                infer_time_ms = (t_infer_end - t_infer_start) * 1000
                
                if action_tensor.ndim != 3:
                    action_tensor = action_tensor.unsqueeze(0)
                action_tensor = action_tensor[:, :actions_per_chunk, :]

                # Postprocess chunk
                _, chunk_size, _ = action_tensor.shape
                processed_actions = []
                for i in range(chunk_size):
                    single_action = action_tensor[:, i, :]
                    processed_action = postprocess(single_action)
                    processed_actions.append(processed_action)
                
                action_chunk = torch.stack(processed_actions, dim=1).squeeze(0).cpu().numpy()
                action_chunk_converted = convert_lerobot_action_to_leisaac(action_chunk)
                for i in range(action_chunk_converted.shape[0]):
                    action_queue.append(torch.from_numpy(action_chunk_converted[i]).to(device))

            # === ENVIRONMENT STEP ===
            step_action = action_queue.popleft()
            
            last_positions = save_positions(env)
            t_step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(step_action.unsqueeze(0))
            t_step_end = time.perf_counter()
            step_time_ms = (t_step_end - t_step_start) * 1000
            
            # --- Update Tracker ---
            tracker.record_timing(infer_time_ms, step_time_ms)
            step_count += 1
            tracker.update_step(step_count)
            
            is_terminated = bool(terminated.item() if isinstance(terminated, torch.Tensor) else terminated)
            is_truncated = bool(truncated.item() if isinstance(truncated, torch.Tensor) else truncated)
            done = is_terminated or is_truncated
            
            if done:
                oranges_in_plate = count_oranges_in_plate(last_positions)
                tracker.end_episode(episode, step_count, is_terminated, oranges_in_plate)

    tracker.print_final_summary(model_id)

except KeyboardInterrupt:
    print("\nForce quitting Isaac Sim...")
except Exception as e:
    print(f"\n❌ CRASH DETECTED: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Closing environment...")
    env.close()