import time
import torch
import numpy as np
from transformers import AutoProcessor
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
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
    EvaluationTracker,
    SubtaskTracker
)

# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "outputs/train/pick-orange-mimic/checkpoints/040000/pretrained_model"
n_episodes = 100
instruction = "Grab orange and place into plate"

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

print(f"Loading trained policy: {model_id}...")
policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config, 
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# ==========================================
# 3. Evaluation Loop
# ==========================================
tracker = EvaluationTracker(n_episodes)
sub_tracker = SubtaskTracker()
print(f"\n--- STARTING EVALUATION: {n_episodes} EPISODES ---")

try:
    for episode in range(n_episodes):
        obs, _ = env.reset()
        policy.reset() 
        
        done = False
        step_count = 0
        tracker.start_episode(episode)
        sub_tracker.reset()

        while not done:
            policy_obs = obs['policy']

            raw_front = policy_obs['front'][0].cpu().numpy()
            raw_wrist = policy_obs['wrist'][0].cpu().numpy()
            save_camera_snapshots(raw_front, raw_wrist, episode, step_count)

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

            # --- Inference ---
            batch = preprocess(obs_frame)
            t_infer_start = time.perf_counter()
            with torch.inference_mode():
                action_output = policy.select_action(batch)
            t_infer_end = time.perf_counter()
            infer_time_ms = (t_infer_end - t_infer_start) * 1000

            # --- Action Processing ---
            action_dict = postprocess(action_output)
            final_action = action_dict.get("action", action_dict) if isinstance(action_dict, dict) else action_dict

            action_np = final_action.cpu().numpy()
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            action_converted = convert_lerobot_action_to_leisaac(action_np)
            step_action = torch.from_numpy(action_converted).to(device)

            # --- Environment Step ---
            last_positions = save_positions(env)
            t_step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(step_action[0].unsqueeze(0))
            t_step_end = time.perf_counter()
            sub_tracker.check_status(env, step_count)
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