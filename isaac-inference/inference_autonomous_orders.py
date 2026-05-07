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
    HomeChecker,
    SubtaskTracker,
    classify_orange_positions,
    count_oranges_in_plate,
    save_positions,
)
from dataset_recorder import SubtaskRecorder


# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "MasterProject2026/Gal-pick-orange-tailedCH20"
n_episodes = 1000
max_steps = 5000

RECORD_ENABLED    = True
RECORD_RESUME     = True   # append to existing partial dataset; False = always start fresh
RECORD_REPO_ID    = "MasterProject2026/Gal-auto-subtasks"
RECORD_LOCAL_PATH = "/home/gal/Documents/MasterProject/isaac-inference/synthetic_datasets/recorded_dataset"

TIMEOUT_STEPS = {
    "GRASP": 700,
    "LIFT":  400,
    "PLACE": 500,
}
RECOVERY_STEPS = 200

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
    if phase == "HOME":
        return "Go back to start position"
    if phase == "RECOVERY":
        return "Go back to start position"
    return "Pick it up"


class OrderController:
    def __init__(self, orange_names):
        self.orange_names = tuple(orange_names)
        self.phase = "SELECT_TARGET"
        self.target_name = None
        self.target_label = None
        self._phase_steps = 0
        self._recovery_remaining = 0
        self._avoid_target = None

    def reset_episode(self):
        self.phase = "SELECT_TARGET"
        self.target_name = None
        self.target_label = None
        self._phase_steps = 0
        self._recovery_remaining = 0
        self._avoid_target = None

    def _set_phase(self, phase: str):
        self.phase = phase
        self._phase_steps = 0

    def _remaining_targets(self, tracker: SubtaskTracker, orange_positions: dict):
        return [name for name in self.orange_names if name not in tracker.placed_oranges and name in orange_positions]

    def _select_target(self, tracker: SubtaskTracker, orange_positions: dict):
        remaining = self._remaining_targets(tracker, orange_positions)
        if not remaining:
            self.target_name = None
            self.target_label = None
            self._set_phase("HOME")
            print("\n🏠 All oranges placed — returning to start position")
            return None

        if self._avoid_target and self._avoid_target not in remaining:
            self._avoid_target = None

        selectable = remaining
        if self._avoid_target in remaining and len(remaining) > 1:
            selectable = [name for name in remaining if name != self._avoid_target]
        elif self._avoid_target in remaining and len(remaining) == 1:
            self._avoid_target = None

        labels = classify_orange_positions(orange_positions)
        self.target_name = random.choice(selectable)
        self.target_label = labels.get(self.target_name, self.target_name)
        self._set_phase("GRASP")
        tracker.reset_display()
        print(f"\n🎯 Selected target: {self.target_name} ({self.target_label})")
        return self.target_name

    def current_prompt(self):
        return build_task_prompt(self.phase, self.target_label)

    def update_after_step(self, tracker: SubtaskTracker, orange_positions: dict, step_count: int):
        if self.phase == "RECOVERY":
            self._recovery_remaining -= 1
            if self._recovery_remaining <= 0:
                self._set_phase("SELECT_TARGET")
            return

        if self.phase == "HOME":
            return

        self._phase_steps += 1
        timeout = TIMEOUT_STEPS.get(self.phase)
        if timeout is not None and self._phase_steps >= timeout:
            print(f"\n⏱  {self.phase} timed out after {self._phase_steps} steps — recovering")
            if self.phase == "GRASP" and self.target_name:
                self._avoid_target = self.target_name
            tracker.reset_grasp_state()
            self.target_name  = None
            self.target_label = None
            self._set_phase("RECOVERY")
            self._recovery_remaining = RECOVERY_STEPS
            return

        if self.target_name is None or self.target_name not in orange_positions or self.target_name in tracker.placed_oranges:
            if self.phase != "SELECT_TARGET":
                self._set_phase("SELECT_TARGET")
            self._select_target(tracker, orange_positions)
            return

        target_z = orange_positions[self.target_name][2].item()

        if self.phase == "GRASP":
            if tracker._grasp_confirmed:
                if tracker.active_orange != self.target_name:
                    print(f"  ℹ️  Grasped {tracker.active_orange} instead of target {self.target_name} — adapting")
                    self.target_name  = tracker.active_orange
                    self.target_label = classify_orange_positions(orange_positions).get(tracker.active_orange, tracker.active_orange)
                self._set_phase("LIFT")
                tracker.reset_display()
                print(f"  ✅ Grasp confirmed for {self.target_name}; switching to lift")
            return

        if self.phase == "LIFT":
            if tracker._lift_confirmed:
                self._set_phase("PLACE")
                tracker.reset_display()
                print(f"  ✅ Lift confirmed for {self.target_name}; switching to place")
                return

            held, _ = tracker._is_orange_held(orange_positions[self.target_name])
            if not held:
                tracker.reset_display()
                print(f"  ⚠️  {self.target_name} fell during lift; returning to grasp")
                tracker.reset_grasp_state()
                tracker.active_orange = self.target_name
                self._set_phase("GRASP")
            return

        if self.phase == "PLACE":
            if self.target_name in tracker.placed_oranges:
                tracker.reset_display()
                print(f"  ✅ Placed {self.target_name}")
                self.target_name = None
                self.target_label = None
                if len(tracker.placed_oranges) >= len(self.orange_names):
                    self._set_phase("HOME")
                    tracker.reset_display()
                    print("\n🏠 All oranges placed — returning to start position")
                else:
                    self._set_phase("SELECT_TARGET")
            elif tracker.active_orange != self.target_name and tracker._place_confirmed:
                self._set_phase("SELECT_TARGET")
            elif self.target_name in orange_positions:
                orange_pos = orange_positions[self.target_name]
                held, _    = tracker._is_orange_held(orange_pos)
                in_plate   = tracker._is_orange_in_plate(orange_pos)
                if not held and not in_plate:
                    tracker.reset_display()
                    print(f"  ⚠️  {self.target_name} fell out during placement; returning to grasp")
                    tracker.reset_grasp_state()
                    tracker.active_orange = self.target_name
                    self._set_phase("GRASP")


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
home_checker     = HomeChecker()
reset_controller = ResetController()
reset_controller.start()

recorder = SubtaskRecorder.create(RECORD_REPO_ID, RECORD_LOCAL_PATH, resume=RECORD_RESUME) if RECORD_ENABLED else None

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
        home_checker.reset()
        controller.reset_episode()
        if recorder:
            recorder.discard()

        done = False
        step_count = 0
        last_model_prompt = None
        last_positions = save_positions(env)

        while not done:
            # --- Reset check ---
            if reset_controller.get_and_clear_reset():
                sub_tracker.reset_display()
                print("\n🔄 Episode reset requested.")
                if recorder:
                    recorder.discard()
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

            # --- Record frame ---
            if recorder:
                recorder.record({
                    "observation.images.front": raw_front,
                    "observation.images.wrist": raw_wrist,
                    "observation.state": joint_pos_converted[0],
                    "action": action_np[0],
                })

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
            elif controller.phase == "HOME":
                home_checker.check(env, step_count)
            elif controller.phase == "RECOVERY":
                pass

            sub_tracker.draw_debug(gripper_tip, jaw_tip, orange_positions)
            phase_before     = controller.phase
            home_fired_before = home_checker._fired
            controller.update_after_step(sub_tracker, orange_positions, step_count)
            phase_after = controller.phase

            # --- Dataset recording: commit, discard, or arm for next phase ---
            if recorder:
                if phase_before == "GRASP" and phase_after == "LIFT":
                    recorder.commit(task=f"Grasp {controller.target_label} orange")
                elif phase_before == "LIFT" and phase_after == "PLACE":
                    recorder.commit(task="Pick it up")
                elif phase_before == "PLACE" and phase_after in ("SELECT_TARGET", "HOME"):
                    recorder.commit(task="Place it into plate")
                elif phase_before == "HOME" and not home_fired_before and home_checker._fired:
                    recorder.commit(task="Go back to start position")
                elif phase_after == "RECOVERY":
                    recorder.discard()

                if phase_before != phase_after and phase_after in ("GRASP", "LIFT", "PLACE", "HOME"):
                    recorder.start()

            # --- Bookkeeping ---
            tracker.record_timing(infer_time_ms, step_time_ms)
            tracker.update_step(step_count)

            is_terminated = tensor_to_bool(terminated)
            is_truncated  = tensor_to_bool(truncated)
            done = is_terminated or is_truncated

            if done:
                if recorder:
                    recorder.discard()
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
    if recorder:
        recorder.finalize()
    print("Closing environment...")
    env.close()