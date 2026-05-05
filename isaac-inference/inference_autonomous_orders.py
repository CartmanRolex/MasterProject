import logging
import random
import threading
import time
from typing import Optional

import numpy as np
import torch
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame

from robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)
from eval_utils import (
    EvaluationTracker,
    SubtaskTracker,
    classify_orange_positions,
    count_oranges_in_plate,
    save_camera_snapshots,
    save_positions,
)


# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "MasterProject2026/Gal-pick-orange-tailedCH20"
n_episodes = 100
max_steps = 5000

dataset_features = {
    "observation.images.front": {"dtype": "video", "shape": (3, 480, 640), "names": ["front"]},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["wrist"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (6,), "names": ["action"]},
}


def tensor_to_bool(value):
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def build_task_prompt(phase: str, label: Optional[str]) -> str:
    if phase == "GRASP":
        return f"Grasp {label} orange" if label else "Grasp orange"
    if phase == "LIFT":
        return "Pick it up"
    if phase == "PLACE":
        return "Place it into plate"
    return "Pick it up"


class OrderController:
    def __init__(self, orange_names):
        self.orange_names = tuple(orange_names)
        self.phase = "SELECT_TARGET"
        self.target_name = None
        self.target_label = None
        self._steps_in_lift = 0
        self._last_target_z = None

    def reset_episode(self):
        self.phase = "SELECT_TARGET"
        self.target_name = None
        self.target_label = None
        self._steps_in_lift = 0
        self._last_target_z = None

    def _remaining_targets(self, tracker: SubtaskTracker, orange_positions: dict):
        return [name for name in self.orange_names if name not in tracker.placed_oranges and name in orange_positions]

    def _select_target(self, tracker: SubtaskTracker, orange_positions: dict):
        remaining = self._remaining_targets(tracker, orange_positions)
        if not remaining:
            self.target_name = None
            self.target_label = None
            self.phase = "DONE"
            return None

        labels = classify_orange_positions(orange_positions)
        self.target_name = random.choice(remaining)
        self.target_label = labels.get(self.target_name, self.target_name)
        self.phase = "GRASP"
        self._steps_in_lift = 0
        self._last_target_z = orange_positions[self.target_name][2].item()
        tracker.reset_display()
        print(f"\n🎯 Selected target: {self.target_name} ({self.target_label})")
        return self.target_name

    def current_prompt(self):
        return build_task_prompt(self.phase, self.target_label)

    def update_after_step(self, tracker: SubtaskTracker, orange_positions: dict, step_count: int):
        if self.phase == "DONE":
            return

        if self.target_name is None or self.target_name not in orange_positions or self.target_name in tracker.placed_oranges:
            if self.phase != "SELECT_TARGET":
                self.phase = "SELECT_TARGET"
            self._select_target(tracker, orange_positions)
            return

        target_z = orange_positions[self.target_name][2].item()

        if self.phase == "GRASP":
            if tracker._grasp_confirmed and tracker.active_orange == self.target_name:
                self.phase = "LIFT"
                self._steps_in_lift = 0
                self._last_target_z = target_z
                tracker.reset_display()
                print(f"  ✅ Grasp confirmed for {self.target_name}; switching to lift")
            return

        if self.phase == "LIFT":
            self._steps_in_lift += 1

            if tracker._lift_confirmed and tracker.active_orange == self.target_name:
                self.phase = "PLACE"
                tracker.reset_display()
                print(f"  ✅ Lift confirmed for {self.target_name}; switching to place")
                return

            fell_during_lift = False
            initial_z = tracker.initial_orange_z.get(self.target_name, target_z)
            if self._last_target_z is not None and target_z < self._last_target_z - 0.004:
                fell_during_lift = True
            if self._steps_in_lift >= 4 and target_z <= initial_z + 0.008:
                fell_during_lift = True

            if fell_during_lift:
                tracker.reset_display()
                print(f"  ⚠️  {self.target_name} fell during lift; returning to grasp")
                tracker.reset_grasp_state()
                tracker.active_orange = self.target_name
                self.phase = "GRASP"
                self._steps_in_lift = 0
                self._last_target_z = target_z
                return

            self._last_target_z = target_z
            return

        if self.phase == "PLACE":
            if self.target_name in tracker.placed_oranges:
                tracker.reset_display()
                print(f"  ✅ Placed {self.target_name}")
                self.target_name = None
                self.target_label = None
                self._steps_in_lift = 0
                self._last_target_z = None
                if len(tracker.placed_oranges) >= len(self.orange_names):
                    self.phase = "DONE"
                    tracker.reset_display()
                    print("\n🏁 All oranges placed")
                else:
                    self.phase = "SELECT_TARGET"
            elif tracker.active_orange != self.target_name and tracker._place_confirmed:
                self.phase = "SELECT_TARGET"


class ResetController:
    def __init__(self):
        self._lock = threading.Lock()
        self._reset_requested = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()

    def get_and_clear_reset(self) -> bool:
        with self._lock:
            flag = self._reset_requested
            self._reset_requested = False
            return flag

    def _listen(self):
        while True:
            try:
                raw = input()
            except EOFError:
                break

            raw = raw.strip().lower()
            if raw in ("r", "reset"):
                with self._lock:
                    self._reset_requested = True


# ==========================================
# 2. Environment & Policy Initialization
# ==========================================
print("Loading LeIsaac Environment...")
envs_dict = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][0].envs[0].unwrapped
env.cfg.episode_length_s = max_steps * env.cfg.sim.dt * env.cfg.decimation

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
sub_tracker = SubtaskTracker(block=False)
controller = OrderController(sub_tracker.orange_names)
reset_controller = ResetController()
reset_controller.start()

logging.getLogger("omni").setLevel(logging.ERROR)
logging.getLogger("carb").setLevel(logging.ERROR)
try:
    import carb

    carb.settings.get_settings().set_string("/log/level", "error")
except ImportError:
    pass

print(f"\n--- STARTING AUTONOMOUS EVALUATION: {n_episodes} EPISODES ---")

try:
    for episode in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()

        tracker.start_episode(episode)
        sub_tracker.reset()
        controller.reset_episode()

        done = False
        step_count = 0
        last_model_prompt = None
        last_positions = save_positions(env)

        while not done:
            # --- Reset check ---
            if reset_controller.get_and_clear_reset():
                sub_tracker.reset_display()
                print("\n🔄 Episode reset requested.")
                oranges_in_plate = count_oranges_in_plate(last_positions)
                tracker.end_episode(episode, step_count, False, oranges_in_plate)
                done = True
                break

            # --- Pre-step: select target + init orange heights (prompt must be ready before inference) ---
            *_, orange_positions = sub_tracker._get_env_data(env)
            if step_count == 0:
                for name, pos in orange_positions.items():
                    sub_tracker.initial_orange_z[name] = pos[2].item()
            if controller.phase == "SELECT_TARGET":
                controller._select_target(sub_tracker, orange_positions)

            # --- Build prompt ---
            task_prompt = controller.current_prompt()
            if task_prompt != last_model_prompt:
                sub_tracker.reset_display()
                print(f"\n{'─' * 50}")
                print(f"  ACTIVE PROMPT → \"{task_prompt}\"")
                print(f"{'─' * 50}\n")
                last_model_prompt = task_prompt

            # --- Observation ---
            policy_obs = obs["policy"]
            raw_front = policy_obs["front"][0].cpu().numpy()
            raw_wrist = policy_obs["wrist"][0].cpu().numpy()
            save_camera_snapshots(raw_front, raw_wrist, episode, step_count)
            joint_pos_converted = convert_leisaac_action_to_lerobot(policy_obs["joint_pos"].cpu().numpy())

            obs_frame = build_inference_frame(
                observation={"front": raw_front, "wrist": raw_wrist, "state": joint_pos_converted[0]},
                ds_features=dataset_features,
                device=device,
                task=task_prompt,
            )

            # --- Inference ---
            batch = preprocess(obs_frame)
            t_infer_start = time.perf_counter()
            with torch.inference_mode():
                action_output = policy.select_action(batch)
            infer_time_ms = (time.perf_counter() - t_infer_start) * 1000

            # --- Action ---
            action_dict = postprocess(action_output)
            final_action = action_dict.get("action", action_dict) if isinstance(action_dict, dict) else action_dict
            action_np = final_action.cpu().numpy()
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            step_action = torch.from_numpy(convert_lerobot_action_to_leisaac(action_np)).to(device)

            # --- Step ---
            last_positions = save_positions(env)
            t_step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(step_action[0].unsqueeze(0))
            step_time_ms = (time.perf_counter() - t_step_start) * 1000
            step_count += 1

            # --- Post-step: single env read for all checks ---
            gripper_tip, jaw_tip, gripper_pos, gripper_force_vec, jaw_force_vec, plate_pos, orange_positions = sub_tracker._get_env_data(env)

            if controller.phase == "GRASP":
                sub_tracker._check_grasp(gripper_tip, jaw_tip, orange_positions, step_count,
                                         gripper_force_vec, jaw_force_vec)
            elif controller.phase == "LIFT":
                sub_tracker._check_lift(gripper_pos, orange_positions, step_count)
            elif controller.phase == "PLACE":
                sub_tracker._check_place(plate_pos, orange_positions, gripper_pos, step_count)

            sub_tracker.draw_debug(gripper_tip, jaw_tip, orange_positions)
            controller.update_after_step(sub_tracker, orange_positions, step_count)

            # --- Bookkeeping ---
            tracker.record_timing(infer_time_ms, step_time_ms)
            tracker.update_step(step_count)

            is_terminated = tensor_to_bool(terminated)
            is_truncated  = tensor_to_bool(truncated)
            done = is_terminated or is_truncated or controller.phase == "DONE"

            if done:
                oranges_in_plate = count_oranges_in_plate(last_positions)
                tracker.end_episode(episode, step_count, is_terminated, oranges_in_plate)

    tracker.print_final_summary(model_id)

except KeyboardInterrupt:
    print("\nForce quitting Isaac Sim...")
except Exception as exc:
    print(f"\n❌ CRASH DETECTED: {exc}")
    import traceback

    traceback.print_exc()
finally:
    print("Closing environment...")
    env.close()