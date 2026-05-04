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

# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "MasterProject2026/Gal-pick-orange-tailedCH20"
n_episodes = 100
MAX_STEPS = 5000

AVAILABLE_PROMPTS = [
    "Pick it up",
    "Place itinto plate",
    "Go back to start position",
]

# ==========================================
# 2. Live Prompt Controller
# ==========================================
class PromptController:
    """
    Runs a background thread that listens for terminal input.

    Usage during simulation:
      Press Enter alone  -> print current prompt and available options
      Type a number      -> switch to that numbered prompt
      Type any text      -> set that text as the custom prompt
      r / reset          -> force-reset the current episode
      t <N>              -> set episode truncation to N steps (updates env directly)
    """
    def __init__(self, initial_prompt: str, prompts: list[str], env):
        self._lock = threading.Lock()
        self._prompt = initial_prompt
        self._prompts = prompts
        self._env = env
        self._reset_requested = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()
        self._print_menu()

    def get(self) -> str:
        with self._lock:
            return self._prompt

    def get_and_clear_reset(self) -> bool:
        with self._lock:
            flag = self._reset_requested
            self._reset_requested = False
            return flag

    def _set(self, new_prompt: str):
        with self._lock:
            self._prompt = new_prompt

    def _current_max_steps(self) -> int:
        return self._env.max_episode_length

    def _set_max_steps(self, n: int):
        step_dt = self._env.cfg.sim.dt * self._env.cfg.decimation
        self._env.cfg.episode_length_s = n * step_dt

    def _print_menu(self):
        print("\n" + "=" * 55)
        print("  LIVE PROMPT SWITCHER")
        print("=" * 55)
        for i, p in enumerate(self._prompts):
            marker = "▶" if p == self._prompt else " "
            print(f"  {marker} [{i}] {p}")
        print("  Enter a number to switch, or type a custom prompt.")
        print(f"  Current: \"{self._prompt}\"")
        print(f"  Truncation: {self._current_max_steps()} steps  (change: t <N>)")
        print("  Reset episode: r")
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

            if raw in ("r", "reset"):
                with self._lock:
                    self._reset_requested = True
                print("\n🔄 Episode reset requested.\n")
                continue

            if raw.startswith("t "):
                parts = raw.split()
                if len(parts) == 2 and parts[1].isdigit():
                    n = int(parts[1])
                    self._set_max_steps(n)
                    print(f"\n✅ Truncation set to {n} steps.\n")
                else:
                    print("\n⚠️  Usage: t <N>  (e.g. t 200)\n")
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
env.cfg.episode_length_s = MAX_STEPS * env.cfg.sim.dt * env.cfg.decimation

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
    env=env,
)
prompt_controller.start()

# ==========================================
# 5. Inference Loop
# ==========================================
print(f"\n--- STARTING: {n_episodes} EPISODES ---")

try:
    for episode in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()

        done = False
        step_count = 0

        while not done:
            instruction = prompt_controller.get()

            policy_obs = obs['policy']

            raw_front = policy_obs['front'][0].cpu().numpy()
            raw_wrist = policy_obs['wrist'][0].cpu().numpy()

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

            with torch.inference_mode():
                action_output = policy.select_action(batch)

            # --- Action Processing ---
            action_dict = postprocess(action_output)
            final_action = action_dict.get("action", action_dict) if isinstance(action_dict, dict) else action_dict

            action_np = final_action.cpu().numpy()
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            action_converted = convert_lerobot_action_to_leisaac(action_np)
            step_action = torch.from_numpy(action_converted).to(device)

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(step_action[0].unsqueeze(0))

            step_count += 1

            is_terminated = bool(terminated.item() if isinstance(terminated, torch.Tensor) else terminated)
            is_truncated = bool(truncated.item() if isinstance(truncated, torch.Tensor) else truncated)
            done = is_terminated or is_truncated or prompt_controller.get_and_clear_reset()

        print(f"Episode {episode + 1} done after {step_count} steps.")

except KeyboardInterrupt:
    print("\nForce quitting Isaac Sim...")
except Exception as e:
    print(f"\n❌ CRASH DETECTED: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Closing environment...")
    env.close()
