import logging
import random
import signal
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
model_id = "MasterProject2026/Gal-auto-subtasks2"

# Number of full robot sessions to run.
# One inference run = env reset → robot picks all oranges → done.
# Each successful subtask within a run produces one subtask recording
# in the dataset (what LeRobot calls an "episode").
n_inference_runs = 100

max_steps = 5000

# --- Dataset recording ---
RECORD_ENABLED      = False
RECORD_RESUME       = True   # True: append to existing dataset  |  False: start fresh (needs RECORD_OVERWRITE)
RECORD_OVERWRITE    = False  # True: delete existing dataset and start fresh (DESTRUCTIVE — set intentionally)
RECORD_DATASET_NAME = "Gal-auto-subtasks2"   # repo → MasterProject2026/<name>, local → synthetic_datasets/<name>/
FREEZE_FRAMES       = 20     # freeze frames appended at the end of each subtask recording

# --- Evaluation metrics ---
EVAL_RESUME          = True   # True: resume completed-run metrics from results/eval_<model>_checkpoint.json
EVAL_CHECKPOINT_PATH = None   # None: use model-specific default in results/

TIMEOUT_STEPS = {
    "GRASP": 700,
    "LIFT":  400,
    "PLACE": 500,
}
RECOVERY_STEPS = 150

dataset_features = {
    "observation.images.front": {"dtype": "video", "shape": (3, 480, 640), "names": ["front"]},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["wrist"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (6,), "names": ["action"]},
}


class DryRunRecorder:
    """Mimics SubtaskRecorder interface but only prints what would be recorded."""

    def __init__(self, freeze_frames: int):
        self._freeze_frames = freeze_frames
        self._armed = False
        self._frame_count = 0

    # Stub so the startup banner line works unchanged.
    class _dataset:
        class meta:
            total_episodes = 0

    def start(self):
        self._armed = True
        self._frame_count = 0

    def record(self, _data):
        if self._armed:
            self._frame_count += 1

    def commit(self, task: str = ""):
        if self._armed:
            print(f"  [dry-run] would record subtask \"{task}\" — "
                  f"{self._frame_count} frames + {self._freeze_frames} frozen frames")
        self._armed = False
        self._frame_count = 0

    def discard(self):
        self._armed = False
        self._frame_count = 0

    def close_writers(self):
        pass

    def push_to_hub(self):
        pass


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
    if phase == "HOME_END":
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
        # Oranges that timed out during GRASP this inference run.
        # Used to avoid wasting time retrying the same orange when the model
        # is deterministic. Only GRASP timeouts count — oranges falling during
        # LIFT or PLACE do NOT go into this set.
        self._grasp_timed_out: set = set()
        # Set to True when all available oranges have been tried and timed out.
        # The inference loop transitions to HOME first (to record the return movement),
        # then resets the episode once home_checker fires.
        self._needs_episode_reset: bool = False

    def reset_episode(self):
        self.phase = "SELECT_TARGET"
        self.target_name = None
        self.target_label = None
        self._phase_steps = 0
        self._recovery_remaining = 0
        self._avoid_target = None
        self._grasp_timed_out = set()
        self._needs_episode_reset = False

    def _set_phase(self, phase: str):
        self.phase = phase
        self._phase_steps = 0

    def _remaining_targets(self, tracker: SubtaskTracker, orange_positions: dict):
        return [name for name in self.orange_names if name not in tracker.placed_oranges and name in orange_positions]

    def _select_target(self, tracker: SubtaskTracker, orange_positions: dict):
        remaining = self._remaining_targets(tracker, orange_positions)

        # Detect oranges already physically in the plate that were never officially confirmed
        # (e.g. dropped during a failed PLACE phase whose recording was discarded).
        for name in list(remaining):
            if tracker._is_orange_in_plate(orange_positions[name]):
                print(f"  ⚠️  {name} already in plate (unrecorded) — excluding from selection")
                tracker.placed_oranges.add(name)
        remaining = [n for n in remaining if n not in tracker.placed_oranges]

        if not remaining:
            self.target_name = None
            self.target_label = None
            self._set_phase("HOME")
            print("\n🏠 All oranges placed — returning to start position")
            return None

        # Only pick from oranges that haven't already timed out during GRASP.
        selectable = [name for name in remaining if name not in self._grasp_timed_out]
        if not selectable:
            # Every unplaced orange has been tried and timed out — flag for reset.
            # The inference loop pre-step will detect this and end the episode.
            self._needs_episode_reset = True
            print("\n⚠️  All oranges have timed out — ending episode")
            return None

        # Short-term single-step avoidance for non-timeout failures (e.g. after a
        # fall during LIFT that sends us back to GRASP on the same orange).
        if self._avoid_target and self._avoid_target not in self._grasp_timed_out:
            if self._avoid_target in selectable and len(selectable) > 1:
                selectable = [name for name in selectable if name != self._avoid_target]
            elif self._avoid_target not in selectable:
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

        if self.phase == "HOME_END":
            self._recovery_remaining -= 1
            if self._recovery_remaining <= 0:
                self._needs_episode_reset = True
                self._set_phase("SELECT_TARGET")
            return

        if self.phase == "HOME":
            return

        self._phase_steps += 1
        timeout = TIMEOUT_STEPS.get(self.phase)
        if timeout is not None and self._phase_steps >= timeout:
            print(f"\n⏱  {self.phase} timed out after {self._phase_steps} steps — recovering")
            if self.phase == "GRASP" and self.target_name:
                self._grasp_timed_out.add(self.target_name)
                self._avoid_target = self.target_name
                untried = [
                    name for name in self.orange_names
                    if name not in tracker.placed_oranges
                    and name not in self._grasp_timed_out
                    and name in orange_positions
                ]
                if untried:
                    print(f"  🔄 {len(untried)} orange(s) still untried — will try another after recovery")
                    tracker.reset_grasp_state()
                    self.target_name  = None
                    self.target_label = None
                    self._set_phase("RECOVERY")
                    self._recovery_remaining = RECOVERY_STEPS
                else:
                    print("  ⚠️  All oranges have been tried — returning home to end episode")
                    tracker.reset_grasp_state()
                    self.target_name  = None
                    self.target_label = None
                    self._set_phase("HOME_END")
                    self._recovery_remaining = RECOVERY_STEPS
            else:
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
        self._stop_requested = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()

    def get_and_clear_reset(self) -> bool:
        with self._lock:
            flag = self._reset_requested
            self._reset_requested = False
            return flag

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

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
            elif raw in ("stop", "s"):
                with self._lock:
                    self._stop_requested = True
                print("\n🛑 Stop requested — will save and push after this episode.")


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
tracker = EvaluationTracker(
    n_inference_runs,
    model_id=model_id,
    checkpoint_path=EVAL_CHECKPOINT_PATH,
    resume=EVAL_RESUME,
)
sub_tracker = SubtaskTracker(block=False)
controller = OrderController(sub_tracker.orange_names)
home_checker     = HomeChecker()
reset_controller = ResetController()
reset_controller.start()

recorder = DryRunRecorder(FREEZE_FRAMES)
if RECORD_ENABLED:
    recorder = SubtaskRecorder.create(
        RECORD_DATASET_NAME,
        resume=RECORD_RESUME,
        overwrite=RECORD_OVERWRITE,
        freeze_frames=FREEZE_FRAMES,
    )

# Install our SIGINT/SIGTERM handler AFTER Isaac Sim has loaded, so it overrides
# Isaac Sim's C++ handler. The handler flushes the dataset and then raises
# KeyboardInterrupt so the normal finally block still saves the evaluation summary.
_writers_closed = False
def _shutdown_handler(_sig, _frame):
    global _writers_closed
    if not _writers_closed:
        _writers_closed = True
        print("\n⚠️  Interrupted — flushing dataset to disk before shutdown...")
        if recorder:
            recorder.close_writers()
    raise KeyboardInterrupt

signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)

logging.getLogger("omni").setLevel(logging.ERROR)
logging.getLogger("carb").setLevel(logging.ERROR)
try:
    import carb

    carb.settings.get_settings().set_string("/log/level", "error")
except ImportError:
    pass

_recorded_so_far = recorder._dataset.meta.total_episodes if recorder else 0
_completed_so_far = len(tracker.episode_records)
print(f"""
{'━' * 52}
  AUTONOMOUS EVALUATION
  Model:                {model_id}
  Inference runs:       {n_inference_runs}
  Completed runs:       {_completed_so_far} already tracked
  Recording:            {'enabled' if recorder else 'disabled'}
  Subtask recordings:   {_recorded_so_far} already saved
{'━' * 52}
""")

upload_after_shutdown = False

try:
    for run_idx in range(tracker.next_episode_index, n_inference_runs):
        print(f"\n{'─' * 52}")
        print(f"  Inference run {run_idx + 1} / {n_inference_runs}")
        print(f"{'─' * 52}")

        obs, _ = env.reset()
        policy.reset()

        tracker.start_episode(run_idx)
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
            # --- Stop check ---
            if reset_controller.stop_requested:
                sub_tracker.reset_display()
                print("\n🛑 Stopping — discarding current episode buffer.")
                if recorder:
                    recorder.discard()
                oranges_in_plate = count_oranges_in_plate(last_positions)
                tracker.end_episode(run_idx, step_count, False, oranges_in_plate)
                done = True
                break

            # --- Reset check ---
            if reset_controller.get_and_clear_reset():
                sub_tracker.reset_display()
                print("\n🔄 Episode reset requested.")
                if recorder:
                    recorder.discard()
                oranges_in_plate = count_oranges_in_plate(last_positions)
                tracker.end_episode(run_idx, step_count, False, oranges_in_plate)
                done = True
                break

            # --- Pre-step: select target + init orange heights (prompt must be ready before inference) ---
            *_, orange_positions = sub_tracker._get_env_data(env)
            if step_count == 1:
                for name, pos in orange_positions.items():
                    sub_tracker.initial_orange_z[name] = pos[2].item()
            if controller.phase == "SELECT_TARGET" and step_count > 0:
                if controller._needs_episode_reset:
                    # All oranges timed out. The RECOVERY already moved the robot home and
                    # was recorded — commit it now as "Go back to start position" and reset.
                    sub_tracker.reset_display()
                    print("\n🔄 All oranges tried — resetting episode.")
                    if recorder:
                        recorder.commit("Go back to start position")
                    oranges_in_plate = count_oranges_in_plate(last_positions)
                    tracker.end_episode(run_idx, step_count, False, oranges_in_plate)
                    torch.cuda.empty_cache()
                    done = True
                    break
                controller._select_target(sub_tracker, orange_positions)
                # Arm the recorder for the GRASP that just started. This must happen here
                # because the SELECT_TARGET→GRASP transition occurs in the pre-step, so
                # phase_before == phase_after == "GRASP" by the time the post-step recording
                # check runs — the normal recorder.start() path would never fire.
                if recorder and controller.phase == "GRASP":
                    recorder.start()

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

            # --- Debug snapshot (step 50 of run 0 only) ---
            if run_idx == 0 and step_count == 50:
                from pathlib import Path
                from PIL import Image
                debug_dir = Path(__file__).parent / "debug"
                debug_dir.mkdir(exist_ok=True)
                for name, img in [("front", raw_front), ("wrist", raw_wrist)]:
                    arr = img.transpose(1, 2, 0) if img.shape[0] == 3 else img
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
                    Image.fromarray(arr).save(debug_dir / f"camera_{name}.png")
                print(f"  📸 Debug snapshots saved to {debug_dir}/")

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

            # Capture state BEFORE any checks mutate it, so commit conditions can
            # detect transitions (e.g. home_fired_before must be False on the step
            # home_checker fires, not True because we just called home_checker.check()).
            phase_before      = controller.phase
            target_before     = controller.target_name
            home_fired_before = home_checker._fired

            if controller.phase == "GRASP":
                sub_tracker._check_grasp(gripper_tip, jaw_tip, orange_positions, step_count,
                                         gripper_force_vec, jaw_force_vec)
            elif controller.phase == "LIFT":
                sub_tracker._check_lift(gripper_pos, orange_positions, step_count)
            elif controller.phase == "PLACE":
                sub_tracker._check_place(plate_pos, orange_positions, gripper_pos, step_count)
            elif controller.phase == "HOME":
                home_checker.check(env, step_count)
            elif controller.phase in ("RECOVERY", "HOME_END"):
                pass

            sub_tracker.draw_debug(gripper_tip, jaw_tip, orange_positions)
            controller.update_after_step(sub_tracker, orange_positions, step_count)
            phase_after = controller.phase

            # --- Dataset recording: commit, discard, or arm for next phase ---
            if recorder:
                if phase_before == "GRASP" and phase_after == "LIFT":
                    recorder.commit(task=f"Grasp {controller.target_label} orange")
                elif phase_before == "LIFT" and phase_after == "PLACE":
                    recorder.commit(task="Pick it up")
                elif phase_before == "PLACE" and target_before in sub_tracker.placed_oranges:
                    # Commit whenever the tracked orange was successfully placed, regardless of
                    # what phase comes next. When more oranges remain, update_after_step() calls
                    # _select_target() inline and jumps straight to GRASP (skipping SELECT_TARGET),
                    # so checking phase_after in ("SELECT_TARGET", "HOME") misses those cases.
                    recorder.commit(task="Place it into plate")
                elif phase_before == "HOME" and not home_fired_before and home_checker._fired:
                    recorder.commit(task="Go back to start position")
                elif phase_before != "RECOVERY" and phase_after == "RECOVERY":
                    recorder.discard()
                elif phase_before != "HOME_END" and phase_after == "HOME_END":
                    recorder.discard()  # drop the failed GRASP frames
                    recorder.start()    # arm fresh for the episode-ending home movement

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
                tracker.end_episode(run_idx, step_count, is_terminated, oranges_in_plate)

        if reset_controller.stop_requested:
            break

    upload_after_shutdown = True

except KeyboardInterrupt:
    print("\nInterrupted — saving evaluation summary and closing writers.")
except Exception as exc:
    print(f"\n❌ CRASH DETECTED: {exc}")
    import traceback
    traceback.print_exc()
finally:
    # Always print and save the evaluation summary, regardless of how the script exits.
    # (finally runs for normal completion, KeyboardInterrupt, and Python exceptions;
    # only os._exit()/SIGKILL can bypass this.)
    tracker.print_final_summary(model_id)
    # Always close parquet writers so every file gets a valid footer,
    # regardless of how the script exits (Ctrl+C, crash, or clean finish).
    if recorder:
        recorder.close_writers()
    print("Closing environment...")
    env.close()

# Push outside the Isaac Sim try/finally so Hub upload errors are visible
# and Isaac Sim shutdown logs don't bury them.
if recorder and RECORD_ENABLED and upload_after_shutdown:
    print("\n📤 Pushing dataset to HuggingFace Hub (this may take a few minutes for video data)...")
    recorder.push_to_hub()
