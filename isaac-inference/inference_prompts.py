import time
import threading
import torch
import numpy as np
from transformers import AutoProcessor
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

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
)

# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "MasterProject2026/pick-orange-mimic-subepisoded"
n_episodes = 100

AVAILABLE_PROMPTS = [
    "Go over an orange",
    "Pick up the orange",
    "Place the orange into plate",
    "Go back to start position",
]

# ==========================================
# 2. Live Prompt Controller
# ==========================================
class PromptController:
    """
    Runs a background thread that listens for terminal input.
    Type a number to switch prompts, or type any custom string directly.

    Usage during simulation:
      Press Enter alone  -> print current prompt and available options
      Type a number      -> switch to that numbered prompt
      Type any text      -> set that text as the custom prompt
    """
    def __init__(self, initial_prompt: str, prompts: list[str]):
        self._lock = threading.Lock()
        self._prompt = initial_prompt
        self._prompts = prompts
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()
        self._print_menu()

    def get(self) -> str:
        with self._lock:
            return self._prompt

    def _set(self, new_prompt: str):
        with self._lock:
            self._prompt = new_prompt

    def _print_menu(self):
        print("\n" + "=" * 55)
        print("  LIVE PROMPT SWITCHER")
        print("=" * 55)
        for i, p in enumerate(self._prompts):
            marker = "▶" if p == self._prompt else " "
            print(f"  {marker} [{i}] {p}")
        print("  Enter a number to switch, or type a custom prompt.")
        print(f"  Current: \"{self._prompt}\"")
        print("=" * 55 + "\n")

    def _listen(self):
        while True:
            try:
                raw = input()
            except EOFError:
                break

            raw = raw.strip()

            if not raw:
                self._print_menu()
                continue

            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(self._prompts):
                    new_prompt = self._prompts[idx]
                    self._set(new_prompt)
                    print(f"\n✅ Prompt switched to: \"{new_prompt}\"\n")
                else:
                    print(f"\n⚠️  Index out of range. Choose 0–{len(self._prompts) - 1}.\n")
            else:
                self._set(raw)
                print(f"\n✅ Custom prompt set: \"{raw}\"\n")


dataset_features = {
    "observation.images.front": {"dtype": "video", "shape": (3, 480, 640), "names": ["front"]},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["wrist"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (6,), "names": ["action"]},
}

# ==========================================
# 3. Environment & Policy Initialization
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
# 4. Start Prompt Controller
# ==========================================
prompt_controller = PromptController(
    initial_prompt=AVAILABLE_PROMPTS[0],
    prompts=AVAILABLE_PROMPTS,
)
prompt_controller.start()

# ==========================================
# 5. Evaluation Loop
# ==========================================
tracker = EvaluationTracker(n_episodes)
print(f"\n--- STARTING EVALUATION: {n_episodes} EPISODES ---")

try:
    for episode in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()

        done = False
        step_count = 0
        tracker.start_episode(episode)

        while not done:
            # Fetch the current prompt (may have changed between steps)
            instruction = prompt_controller.get()

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
                task=instruction  # live prompt injected here
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