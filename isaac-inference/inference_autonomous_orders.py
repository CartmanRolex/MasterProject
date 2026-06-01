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
    EpisodeStory,
    HomeChecker,
    SubtaskTracker,
    classify_orange_positions,
    count_oranges_in_plate,
    save_positions,
)
from dataset_recorder import SubtaskRecorder, SYNTHETIC_DATASETS_DIR, merge_staging_into


# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "MasterProject2026/Gal-merged-tailed-auto"

# Number of full robot sessions to run.
# One inference run = env reset → robot picks all oranges → done.
# Each successful subtask within a run produces one subtask recording
# in the dataset (what LeRobot calls an "episode").
n_inference_runs = 500

max_steps = 5000

# --- Dataset recording ---ss
RECORD_ENABLED      = False
RECORD_RESUME       = True   # True: append to existing dataset  |  False: start fresh (needs RECORD_OVERWRITE)
RECORD_OVERWRITE    = False  # True: delete existing dataset and start fresh (DESTRUCTIVE — set intentionally)
RECORD_DATASET_NAME = "-"   # repo → MasterProject2026/<name>, local → synthetic_datasets/<name>/
FREEZE_FRAMES       = 20     # freeze frames appended at the end of each subtask recording

# --- Full-success data generation ---
# When True: only record episodes where all 3 oranges are successfully placed.
#   - n_inference_runs counts fully successful episodes (not attempts).
#   - Subtasks are written to a staging dataset and merged into the real dataset
#     only on full success — failed attempts produce zero disk writes.
#   - On the 2nd grasp failure for any orange the episode aborts immediately.
#   - EvaluationTracker is not used (evaluation stats are unaffected).
FULL_SUCCESS_DATA_GENERATION = False

# --- Evaluation metrics ---
EVAL_RESUME          = True  # True: resume completed-run metrics from results/<model>/checkpoint.json
EVAL_CHECKPOINT_PATH = None   # None: use model-specific default in results/

TIMEOUT_STEPS = {
    "GRASP": 700,
    "LIFT":  400,
    "PLACE": 500,
}

# --- Debug visualization ---
DEBUG_DRAW_PLATE_BOUNDS = False    # Draw the COM/radius/z plate occupancy cylinder in Isaac viewport

# --- Scripted spatial reset ---
SCRIPTED_SPATIAL_RESET     = True  # False → fall back to VLA "Go back to start position" prompt
SHOULDER_LIFT_JOINT_INDEX = 1      # joint order: shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper
SPATIAL_RESET_SHOULDER_LIFT_STEPS = 40  # first move only shoulder_lift to the episode-start value
SPATIAL_RESET_FULL_STEPS = 60      # then move all joints to the episode-start pose
SPATIAL_RESET_STEPS = SPATIAL_RESET_SHOULDER_LIFT_STEPS + SPATIAL_RESET_FULL_STEPS
SPATIAL_RESET_INTERP_STEPS = SPATIAL_RESET_FULL_STEPS

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

    def commit(self, task: str = "", **_):
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
    # These prompts are only reached when SCRIPTED_SPATIAL_RESET = False.
    # When scripted reset is active the VLA is bypassed entirely for these phases.
    if phase == "SPATIAL_RESET":
        return "Go back to start position"
    if phase == "ABORT_HOME":
        return "Go back to start position"
    return "Pick it up"


def is_spatial_reset_shoulder_lift_stage(controller) -> bool:
    return (
        SCRIPTED_SPATIAL_RESET
        and SPATIAL_RESET_SHOULDER_LIFT_STEPS > 0
        and controller.phase in ("SPATIAL_RESET", "ABORT_HOME")
        and controller._spatial_reset_remaining > SPATIAL_RESET_FULL_STEPS
    )


def spatial_reset_shoulder_lift_pose(start_joint, home_joint, progress):
    pose = start_joint.copy()
    idx = SHOULDER_LIFT_JOINT_INDEX
    pose[idx] = (1.0 - progress) * start_joint[idx] + progress * home_joint[idx]
    return pose


def exhausted_oranges(controller):
    return {name for name, count in controller._timeout_count.items() if count >= 2}


def finish_story_episode(story, step_count, oranges_in_plate, end_reason,
                         is_success, plate_pos, orange_positions,
                         sub_tracker, controller):
    if story is None:
        return None
    # Build the recorded placed_oranges purely from the final geometric snapshot so
    # it agrees with oranges_in_plate. Real-time placed_oranges credit is intentionally
    # not seeded here — an orange that bounced out must not be recorded as placed.
    placed_oranges = set()
    for name, pos in orange_positions.items():
        if sub_tracker._is_orange_in_plate(pos):
            placed_oranges.add(name)
    return story.build_record(
        step_count=step_count,
        oranges_in_plate=oranges_in_plate,
        end_reason=end_reason,
        is_success=is_success,
        plate_pos=plate_pos,
        orange_positions=orange_positions,
        placed_oranges=placed_oranges,
        abandoned_oranges=exhausted_oranges(controller),
    )


def fallback_scene(last_plate_pos, last_orange_positions, last_positions, orange_names):
    plate_pos = last_plate_pos
    orange_positions = last_orange_positions
    if plate_pos is None and last_positions:
        plate_pos = last_positions.get("plate")
    if not orange_positions and last_positions:
        orange_positions = {
            name: last_positions[name]
            for name in orange_names
            if name in last_positions
        }
    return plate_pos, orange_positions


def orchestrated_oranges_in_plate(positions, sub_tracker):
    # Pure geometric snapshot of the final frame — must agree with what is
    # actually in the (possibly tilted) plate. placed_oranges is a real-time
    # control signal only and must not inflate the final count, otherwise an
    # orange that bounced out after being confirmed stays counted forever.
    return count_oranges_in_plate(positions)


class OrderController:
    def __init__(self, orange_names):
        self.orange_names = tuple(orange_names)
        self.phase = "SELECT_TARGET"
        self.target_name = None
        self.target_label = None
        self._phase_steps = 0
        self._spatial_reset_remaining = 0
        # Per-orange timeout counts (any phase). An orange is retried once (spatial reset +
        # GRASP from scratch) after its first timeout; abandoned after the second.
        self._timeout_count: dict = {}
        # Orange queued for a retry GRASP after the current spatial reset completes.
        # Set on first timeout; cleared once _select_target force-selects it.
        self._retry_target: Optional[str] = None
        # Set to True when all oranges are exhausted. The inference loop ends the episode.
        self._needs_episode_reset: bool = False
        # Mechanism counters — reset each episode, passed to EvaluationTracker.end_episode().
        self.n_local_retries = 0   # orange slipped during LIFT/PLACE → retry same orange (no spatial reset)
        self.n_redirections  = 0   # target abandoned after 2nd timeout → spatial reset → new orange

    def reset_episode(self):
        self.phase = "SELECT_TARGET"
        self.target_name = None
        self.target_label = None
        self._phase_steps = 0
        self._spatial_reset_remaining = 0
        self._timeout_count = {}
        self._retry_target = None
        self._needs_episode_reset = False
        self.n_local_retries = 0
        self.n_redirections  = 0

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

        # Oranges that have exhausted both attempts (2 timeouts on any phase).
        exhausted = {name for name, count in self._timeout_count.items() if count >= 2}
        selectable = [name for name in remaining if name not in exhausted]
        if not selectable:
            self._needs_episode_reset = True
            print("\n⚠️  All oranges exhausted — ending episode")
            return None

        labels = classify_orange_positions(orange_positions)

        # If a retry is queued (first timeout → spatial reset → retry same orange),
        # force-select that orange rather than picking randomly.
        if self._retry_target and self._retry_target in selectable:
            retry = self._retry_target
            self._retry_target = None
            self.target_name = retry
            self.target_label = labels.get(retry, retry)
            self._set_phase("GRASP")
            tracker.reset_display()
            print(f"\n🎯 Retrying target: {retry} ({self.target_label}) after spatial reset")
            return retry
        self._retry_target = None  # clear if the queued orange is no longer selectable

        self.target_name = random.choice(selectable)
        self.target_label = labels.get(self.target_name, self.target_name)
        self._set_phase("GRASP")
        tracker.reset_display()
        print(f"\n🎯 Selected target: {self.target_name} ({self.target_label})")
        return self.target_name

    def current_prompt(self):
        return build_task_prompt(self.phase, self.target_label)

    def update_after_step(self, tracker: SubtaskTracker, orange_positions: dict, step_count: int):
        if self.phase == "SPATIAL_RESET":
            self._spatial_reset_remaining -= 1
            if self._spatial_reset_remaining <= 0:
                self._set_phase("SELECT_TARGET")
            return

        if self.phase == "ABORT_HOME":
            self._spatial_reset_remaining -= 1
            if self._spatial_reset_remaining <= 0:
                self._needs_episode_reset = True
                self._set_phase("SELECT_TARGET")
            return

        if self.phase == "HOME":
            return

        self._phase_steps += 1
        timeout = TIMEOUT_STEPS.get(self.phase)
        if timeout is not None and self._phase_steps >= timeout:
            timed_out_orange = self.target_name
            if timed_out_orange:
                count = self._timeout_count.get(timed_out_orange, 0) + 1
                self._timeout_count[timed_out_orange] = count
            else:
                count = 2  # no active target → treat as exhausted, go straight to redirect

            tracker.reset_grasp_state()
            self.target_name  = None
            self.target_label = None

            if count == 1:
                # First timeout on this orange: spatial reset then retry GRASP from scratch.
                print(f"\n⏱  {self.phase} timed out after {self._phase_steps} steps")
                print(f"  🔁 Local retry — spatial reset then retry GRASP for {timed_out_orange}")
                self._retry_target = timed_out_orange
                self.n_local_retries += 1
                self._set_phase("SPATIAL_RESET")
                self._spatial_reset_remaining = SPATIAL_RESET_STEPS
            else:
                if timed_out_orange:
                    print(f"\n⏱  {self.phase} timed out after {self._phase_steps} steps (2nd attempt — abandoning {timed_out_orange})")
                else:
                    print(f"\n⏱  {self.phase} timed out after {self._phase_steps} steps")

                # In full-success mode any exhausted orange means we can never place
                # all 3 — abort the episode immediately without ABORT_HOME.
                if FULL_SUCCESS_DATA_GENERATION:
                    print("  ⛔ Full-success mode — aborting episode immediately")
                    self._needs_episode_reset = True
                    self._set_phase("SELECT_TARGET")
                    return

                # Second timeout: permanently abandon this orange, redirect to another.
                exhausted = {name for name, c in self._timeout_count.items() if c >= 2}
                available = [
                    name for name in self.orange_names
                    if name not in tracker.placed_oranges
                    and name not in exhausted
                    and name in orange_positions
                ]
                if available:
                    print(f"  🔀 Target redirection — {len(available)} orange(s) still available")
                    print(f"  🏠 Spatial reset — moving to home (precondition for target change)")
                    self.n_redirections += 1
                    self._set_phase("SPATIAL_RESET")
                    self._spatial_reset_remaining = SPATIAL_RESET_STEPS
                else:
                    print("  ⚠️  All oranges exhausted — initiating spatial reset before episode abort")
                    self.n_redirections += 1
                    self._set_phase("ABORT_HOME")
                    self._spatial_reset_remaining = SPATIAL_RESET_STEPS
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
                print(f"  🔁 Local retry — {self.target_name} fell during LIFT, returning to GRASP same orange")
                tracker.reset_grasp_state()
                tracker.active_orange = self.target_name
                self.n_local_retries += 1
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
                    print(f"  🔁 Local retry — {self.target_name} fell during PLACE, returning to GRASP same orange")
                    tracker.reset_grasp_state()
                    tracker.active_orange = self.target_name
                    self.n_local_retries += 1
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
sub_tracker.DEBUG_DRAW_PLATE_BOUNDS = DEBUG_DRAW_PLATE_BOUNDS
controller = OrderController(sub_tracker.orange_names)
home_checker     = HomeChecker()
reset_controller = ResetController()
reset_controller.start()

recorder = None
if RECORD_ENABLED:
    if FULL_SUCCESS_DATA_GENERATION:
        # Ensure the main dataset exists; staging is created fresh per episode attempt.
        _main_recorder = SubtaskRecorder.create(
            RECORD_DATASET_NAME,
            resume=RECORD_RESUME,
            overwrite=RECORD_OVERWRITE,
            freeze_frames=FREEZE_FRAMES,
        )
        _main_recorder.close_writers()
        # recorder is reassigned at the start of each episode attempt in the loop below
    else:
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

if FULL_SUCCESS_DATA_GENERATION:
    _recorded_so_far = _main_recorder._dataset.meta.total_episodes if RECORD_ENABLED else 0
    print(f"""
{'━' * 52}
  FULL-SUCCESS DATA GENERATION
  Model:                {model_id}
  Target successes:     {n_inference_runs}
  Subtask recordings:   {_recorded_so_far} already saved
{'━' * 52}
""")
else:
    _recorded_so_far = recorder._dataset.meta.total_episodes if RECORD_ENABLED else 0
    _completed_so_far = len(tracker.episode_records)
    print(f"""
{'━' * 52}
  AUTONOMOUS EVALUATION
  Model:                {model_id}
  Inference runs:       {n_inference_runs}
  Completed runs:       {_completed_so_far} already tracked
  Recording:            {'enabled' if RECORD_ENABLED else 'disabled'}
  Subtask recordings:   {_recorded_so_far} already saved
{'━' * 52}
""")

upload_after_shutdown = False

try:
    _fs_successes = 0   # full-success mode counter
    _fs_attempts  = 0

    _fs_active = FULL_SUCCESS_DATA_GENERATION  # shorthand used throughout the loop

    # In full-success mode run_idx is an unlimited attempt counter; the real
    # termination condition is _fs_successes >= n_inference_runs checked below.
    for run_idx in (range(10 ** 9) if _fs_active else range(tracker.next_episode_index, n_inference_runs)):
        if _fs_active:
            _fs_attempts += 1
            print(f"\n{'─' * 52}")
            print(f"  Attempt {_fs_attempts}  |  successes {_fs_successes} / {n_inference_runs}")
            print(f"{'─' * 52}")
            if RECORD_ENABLED:
                recorder = SubtaskRecorder.create(
                    RECORD_DATASET_NAME + "_staging",
                    resume=False, overwrite=True, freeze_frames=FREEZE_FRAMES,
                )
        else:
            print(f"\n{'─' * 52}")
            print(f"  Inference run {run_idx + 1} / {n_inference_runs}")
            print(f"{'─' * 52}")

        obs, _ = env.reset()
        policy.reset()

        if not _fs_active:
            tracker.start_episode(run_idx)
        sub_tracker.reset()
        home_checker.reset()
        controller.reset_episode()
        if recorder:
            recorder.discard()

        episode_succeeded = False  # full-success mode only

        done = False
        step_count = 0
        last_model_prompt = None
        last_positions = save_positions(env)
        last_plate_pos = None
        last_orange_positions = {}
        last_camera_images = None
        episode_start_joint_pos   = None  # arm pose at episode start (LeIsaac radians); scripted reset target
        spatial_reset_start_joint = None  # arm pose when SPATIAL_RESET/ABORT_HOME begins
        episode_story = EpisodeStory(run_idx, model_id) if not _fs_active else None
        story_initial_scene_recorded = False

        while not done:
            # --- Stop check ---
            if reset_controller.stop_requested:
                sub_tracker.reset_display()
                print("\n🛑 Stopping — discarding current episode buffer.")
                if recorder:
                    recorder.discard()
                if not _fs_active:
                    oranges_in_plate = orchestrated_oranges_in_plate(last_positions, sub_tracker)
                    plate_pos, orange_positions_for_story = fallback_scene(
                        last_plate_pos, last_orange_positions, last_positions, controller.orange_names
                    )
                    story_record = finish_story_episode(
                        episode_story, step_count, oranges_in_plate, "manual_stop", False,
                        plate_pos, orange_positions_for_story, sub_tracker, controller
                    )
                    tracker.end_episode(run_idx, step_count, False, oranges_in_plate,
                                        n_local_retries=controller.n_local_retries,
                                        n_redirections=controller.n_redirections,
                                        n_oranges_abandoned=sum(1 for c in controller._timeout_count.values() if c >= 2),
                                        camera_images=last_camera_images,
                                        episode_story=story_record)
                done = True
                break

            # --- Reset check ---
            if reset_controller.get_and_clear_reset():
                sub_tracker.reset_display()
                print("\n🔄 Episode reset requested.")
                if recorder:
                    recorder.discard()
                if not _fs_active:
                    oranges_in_plate = orchestrated_oranges_in_plate(last_positions, sub_tracker)
                    plate_pos, orange_positions_for_story = fallback_scene(
                        last_plate_pos, last_orange_positions, last_positions, controller.orange_names
                    )
                    story_record = finish_story_episode(
                        episode_story, step_count, oranges_in_plate, "manual_reset", False,
                        plate_pos, orange_positions_for_story, sub_tracker, controller
                    )
                    tracker.end_episode(run_idx, step_count, False, oranges_in_plate,
                                        n_local_retries=controller.n_local_retries,
                                        n_redirections=controller.n_redirections,
                                        n_oranges_abandoned=sum(1 for c in controller._timeout_count.values() if c >= 2),
                                        camera_images=last_camera_images,
                                        episode_story=story_record)
                done = True
                break

            # --- Pre-step: select target + init orange heights (prompt must be ready before inference) ---
            _, _, _, _, _, plate_pos, orange_positions = sub_tracker._get_env_data(env)
            last_plate_pos = plate_pos
            last_orange_positions = orange_positions
            if step_count == 1:
                # Step 0 still has stale Isaac Sim buffers from the previous episode.
                # Heights and _select_target are deferred to step 1 (first valid scene state).
                for name, pos in orange_positions.items():
                    sub_tracker.initial_orange_z[name] = pos[2].item()
                if episode_story and not story_initial_scene_recorded:
                    episode_story.record_initial_scene(step_count, plate_pos, orange_positions)
                    story_initial_scene_recorded = True
            if controller.phase == "SELECT_TARGET" and step_count > 0:
                if controller._needs_episode_reset:
                    sub_tracker.reset_display()
                    if episode_story:
                        episode_story.add_event(
                            step_count,
                            "episode_reset_needed",
                            phase="SELECT_TARGET",
                            outcome="failure",
                            reason="all_targets_exhausted",
                            abandoned_oranges=sorted(exhausted_oranges(controller)),
                        )
                    if _fs_active:
                        # Immediate abort — discard staging, no HOME commit.
                        print("\n⛔ Aborting episode (full-success mode).")
                        if recorder:
                            recorder.discard()
                    else:
                        # Normal mode: commit HOME movement then end episode.
                        print("\n🔄 All oranges tried — resetting episode.")
                        if recorder:
                            recorder.commit("Go back to start position",
                                           n_placed=len(sub_tracker.placed_oranges))
                        oranges_in_plate = orchestrated_oranges_in_plate(last_positions, sub_tracker)
                        story_record = finish_story_episode(
                            episode_story, step_count, oranges_in_plate, "all_targets_exhausted", False,
                            plate_pos, orange_positions, sub_tracker, controller
                        )
                        tracker.end_episode(run_idx, step_count, False, oranges_in_plate,
                                            n_local_retries=controller.n_local_retries,
                                            n_redirections=controller.n_redirections,
                                            n_oranges_abandoned=sum(1 for c in controller._timeout_count.values() if c >= 2),
                                            camera_images=last_camera_images,
                                            episode_story=story_record)
                    torch.cuda.empty_cache()
                    done = True
                    break
                retry_target_before = controller._retry_target
                controller._select_target(sub_tracker, orange_positions)
                if episode_story and controller.phase == "GRASP":
                    reason = "retry_after_timeout" if retry_target_before == controller.target_name else "new_target"
                    episode_story.add_event(
                        step_count,
                        "target_selected",
                        phase="GRASP",
                        requested_orange=controller.target_name,
                        requested_label=controller.target_label,
                        outcome="selected",
                        reason=reason,
                    )
                    episode_story.start_attempt(
                        step_count,
                        "GRASP",
                        controller.current_prompt(),
                        len(sub_tracker.placed_oranges),
                        controller.target_name,
                        controller.target_label,
                    )
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
                if episode_story:
                    episode_story.add_event(
                        step_count,
                        "prompt_changed",
                        phase=controller.phase,
                        requested_orange=controller.target_name,
                        requested_label=controller.target_label,
                        outcome="prompt_active",
                        prompt=task_prompt,
                    )
                last_model_prompt = task_prompt

            # --- Observation ---
            policy_obs = obs["policy"]
            raw_front = policy_obs["front"][0].cpu().numpy()
            raw_wrist = policy_obs["wrist"][0].cpu().numpy()
            last_camera_images = {"front": raw_front, "wrist": raw_wrist}

            # Save episode-start joint pose (scripted reset target). Deferred to step 1
            # for the same reason as height init: step 0 buffers are stale post-reset.
            if step_count == 1 and SCRIPTED_SPATIAL_RESET:
                episode_start_joint_pos = policy_obs["joint_pos"][0].cpu().numpy()

            # Snapshot arm pose on the first step of reset; stage 1 moves only
            # shoulder_lift away from the oranges, stage 2 moves all joints home.
            if (SCRIPTED_SPATIAL_RESET
                    and controller.phase in ("SPATIAL_RESET", "ABORT_HOME")
                    and controller._spatial_reset_remaining == SPATIAL_RESET_STEPS):
                spatial_reset_start_joint = policy_obs["joint_pos"][0].cpu().numpy()

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

            # --- Inference (VLA) or scripted spatial reset ---
            if (SCRIPTED_SPATIAL_RESET
                    and controller.phase in ("SPATIAL_RESET", "ABORT_HOME")
                    and episode_start_joint_pos is not None
                    and spatial_reset_start_joint is not None):
                if is_spatial_reset_shoulder_lift_stage(controller):
                    steps_elapsed = SPATIAL_RESET_STEPS - controller._spatial_reset_remaining
                    t = min(steps_elapsed / SPATIAL_RESET_SHOULDER_LIFT_STEPS, 1.0)
                    scripted_joint = spatial_reset_shoulder_lift_pose(
                        spatial_reset_start_joint,
                        episode_start_joint_pos,
                        t,
                    )
                else:
                    shoulder_lift_pose = spatial_reset_shoulder_lift_pose(
                        spatial_reset_start_joint,
                        episode_start_joint_pos,
                        1.0,
                    )
                    steps_elapsed = SPATIAL_RESET_FULL_STEPS - controller._spatial_reset_remaining
                    t = min(steps_elapsed / SPATIAL_RESET_INTERP_STEPS, 1.0)
                    scripted_joint = (1.0 - t) * shoulder_lift_pose + t * episode_start_joint_pos
                step_action = torch.from_numpy(scripted_joint[None, :]).to(device)
                action_np   = convert_leisaac_action_to_lerobot(scripted_joint[None, :])
                infer_time_ms = 0.0
            else:
                obs_frame = build_inference_frame(
                    observation={"front": raw_front, "wrist": raw_wrist, "state": joint_pos_converted[0]},
                    ds_features=dataset_features,
                    device=device,
                    task=task_prompt,
                )
                batch = preprocess(obs_frame)
                t_infer_start = time.perf_counter()
                with torch.inference_mode():
                    action_output = policy.select_action(batch)
                infer_time_ms = (time.perf_counter() - t_infer_start) * 1000
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
            label_before      = controller.target_label
            active_before     = sub_tracker.active_orange
            placed_before     = set(sub_tracker.placed_oranges)
            retries_before    = controller.n_local_retries
            redirects_before  = controller.n_redirections
            exhausted_before  = exhausted_oranges(controller)
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
            elif controller.phase in ("SPATIAL_RESET", "ABORT_HOME"):
                pass

            sub_tracker.draw_debug(gripper_tip, jaw_tip, orange_positions)
            controller.update_after_step(sub_tracker, orange_positions, step_count)
            phase_after = controller.phase
            placed_after = set(sub_tracker.placed_oranges)
            exhausted_after = exhausted_oranges(controller)

            if phase_before not in ("SPATIAL_RESET", "ABORT_HOME") and phase_after in ("SPATIAL_RESET", "ABORT_HOME"):
                policy.reset()
                spatial_reset_start_joint = None

            if episode_story:
                if phase_before in ("GRASP", "LIFT", "PLACE") and phase_after != phase_before:
                    actual_orange = None
                    result = "failure"
                    failure_reason = None

                    if phase_before == "GRASP":
                        actual_orange = sub_tracker.active_orange or controller.target_name
                        if phase_after == "LIFT":
                            result = "success"
                            if target_before and actual_orange and target_before != actual_orange:
                                failure_reason = "wrong_orange"
                                episode_story.add_event(
                                    step_count,
                                    "wrong_orange_grasped",
                                    phase="GRASP",
                                    requested_orange=target_before,
                                    requested_label=label_before,
                                    actual_orange=actual_orange,
                                    outcome="success",
                                    reason="requested_orange_did_not_match_grasped_orange",
                                )
                            episode_story.add_event(
                                step_count,
                                "grasp_success",
                                phase="GRASP",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="success",
                                reason=failure_reason,
                            )
                        elif phase_after in ("SPATIAL_RESET", "ABORT_HOME"):
                            result = "timeout"
                            failure_reason = "timeout"
                            episode_story.add_event(
                                step_count,
                                "grasp_timeout",
                                phase="GRASP",
                                requested_orange=target_before,
                                requested_label=label_before,
                                outcome="timeout",
                                reason="timeout",
                            )
                        else:
                            failure_reason = "phase_changed"

                    elif phase_before == "LIFT":
                        actual_orange = active_before or target_before
                        if phase_after == "PLACE":
                            result = "success"
                            episode_story.add_event(
                                step_count,
                                "lift_success",
                                phase="LIFT",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="success",
                            )
                        elif phase_after == "GRASP":
                            failure_reason = "dropped_during_lift"
                            episode_story.add_event(
                                step_count,
                                "orange_dropped",
                                phase="LIFT",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="failure",
                                reason=failure_reason,
                            )
                            episode_story.add_event(
                                step_count,
                                "lift_failure",
                                phase="LIFT",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="failure",
                                reason=failure_reason,
                            )
                        elif phase_after in ("SPATIAL_RESET", "ABORT_HOME"):
                            result = "timeout"
                            failure_reason = "timeout"
                            episode_story.add_event(
                                step_count,
                                "lift_timeout",
                                phase="LIFT",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="timeout",
                                reason="timeout",
                            )
                        else:
                            failure_reason = "phase_changed"

                    elif phase_before == "PLACE":
                        newly_placed = sorted(placed_after - placed_before)
                        actual_orange = newly_placed[0] if newly_placed else target_before
                        if target_before in placed_after or newly_placed:
                            result = "success"
                            if target_before and actual_orange and target_before != actual_orange:
                                failure_reason = "wrong_orange"
                            episode_story.add_event(
                                step_count,
                                "place_success",
                                phase="PLACE",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="success",
                                reason=failure_reason,
                            )
                        elif phase_after == "GRASP":
                            failure_reason = "dropped_during_place"
                            episode_story.add_event(
                                step_count,
                                "orange_dropped",
                                phase="PLACE",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="failure",
                                reason=failure_reason,
                            )
                            episode_story.add_event(
                                step_count,
                                "place_failure",
                                phase="PLACE",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="failure",
                                reason=failure_reason,
                            )
                        elif phase_after in ("SPATIAL_RESET", "ABORT_HOME"):
                            result = "timeout"
                            failure_reason = "timeout"
                            episode_story.add_event(
                                step_count,
                                "place_timeout",
                                phase="PLACE",
                                requested_orange=target_before,
                                requested_label=label_before,
                                actual_orange=actual_orange,
                                outcome="timeout",
                                reason="timeout",
                            )
                        else:
                            failure_reason = "phase_changed"

                    episode_story.finish_attempt(
                        step_count,
                        result=result,
                        actual_orange=actual_orange,
                        failure_reason=failure_reason,
                    )

                if controller.n_local_retries > retries_before:
                    if phase_before in ("LIFT", "PLACE") and phase_after == "GRASP":
                        retry_reason = f"dropped_during_{phase_before.lower()}"
                    elif phase_after == "SPATIAL_RESET":
                        retry_reason = "timeout"
                    else:
                        retry_reason = "local_retry"
                    episode_story.add_event(
                        step_count,
                        "local_retry",
                        phase=phase_after,
                        requested_orange=target_before or controller.target_name,
                        requested_label=label_before or controller.target_label,
                        actual_orange=active_before or target_before,
                        outcome="retry_same_target",
                        reason=retry_reason,
                    )

                abandoned_now = sorted(exhausted_after - exhausted_before)
                for abandoned in abandoned_now:
                    episode_story.add_event(
                        step_count,
                        "orange_abandoned",
                        phase=phase_after,
                        requested_orange=abandoned,
                        requested_label=label_before if abandoned == target_before else None,
                        actual_orange=abandoned,
                        outcome="abandoned",
                        reason="second_timeout",
                    )

                if controller.n_redirections > redirects_before:
                    available = [
                        name for name in controller.orange_names
                        if name not in placed_after
                        and name not in exhausted_after
                        and name in orange_positions
                    ]
                    episode_story.add_event(
                        step_count,
                        "target_redirection",
                        phase=phase_after,
                        requested_orange=target_before,
                        requested_label=label_before,
                        outcome="redirect",
                        reason="target_abandoned_after_repeated_failure",
                        available_oranges=available,
                    )

                if phase_before not in ("SPATIAL_RESET", "ABORT_HOME") and phase_after in ("SPATIAL_RESET", "ABORT_HOME"):
                    episode_story.add_event(
                        step_count,
                        "spatial_reset",
                        phase=phase_after,
                        requested_orange=target_before,
                        requested_label=label_before,
                        outcome="started",
                        reason="target_change_precondition" if controller.n_redirections > redirects_before else "retry_after_timeout",
                    )
                elif phase_before in ("SPATIAL_RESET", "ABORT_HOME") and phase_after == "SELECT_TARGET":
                    episode_story.add_event(
                        step_count,
                        "spatial_reset_finished",
                        phase=phase_after,
                        outcome="finished",
                        reason="abort_home_complete" if phase_before == "ABORT_HOME" else "reset_complete",
                    )

                if phase_before != phase_after and phase_after in ("GRASP", "LIFT", "PLACE"):
                    episode_story.start_attempt(
                        step_count,
                        phase_after,
                        build_task_prompt(phase_after, controller.target_label),
                        len(sub_tracker.placed_oranges),
                        controller.target_name,
                        controller.target_label,
                    )

            # End the episode immediately when the third orange is placed.
            # FULL_SUCCESS mode additionally flips episode_succeeded inside the
            # recorder block below for staging-merge bookkeeping.
            if (phase_before == "PLACE"
                    and len(sub_tracker.placed_oranges) == len(controller.orange_names)):
                done = True

            # After a scripted spatial reset, flush the VLA action queue so stale
            # pre-reset actions don't replay on the first GRASP step.
            if (SCRIPTED_SPATIAL_RESET
                    and phase_before in ("SPATIAL_RESET", "ABORT_HOME")
                    and phase_after == "SELECT_TARGET"):
                policy.reset()

            # --- Dataset recording: commit, discard, or arm for next phase ---
            if recorder:
                if phase_before == "GRASP" and phase_after == "LIFT":
                    recorder.commit(task=f"Grasp {controller.target_label} orange",
                                   n_placed=len(sub_tracker.placed_oranges))
                elif phase_before == "LIFT" and phase_after == "PLACE":
                    recorder.commit(task="Pick it up",
                                   n_placed=len(sub_tracker.placed_oranges))
                elif phase_before == "PLACE" and target_before in sub_tracker.placed_oranges:
                    # Commit whenever the tracked orange was successfully placed, regardless of
                    # what phase comes next. When more oranges remain, update_after_step() calls
                    # _select_target() inline and jumps straight to GRASP (skipping SELECT_TARGET),
                    # so checking phase_after in ("SELECT_TARGET", "HOME") misses those cases.
                    recorder.commit(task="Place it into plate",
                                   n_placed=len(sub_tracker.placed_oranges))
                    # Full-success: all oranges placed → merge staging on episode end.
                    # (done = True is already set above for both modes.)
                    if (_fs_active
                            and len(sub_tracker.placed_oranges) == len(controller.orange_names)):
                        episode_succeeded = True
                elif phase_before == "HOME" and not home_fired_before and home_checker._fired:
                    recorder.commit(task="Go back to start position",
                                   n_placed=len(sub_tracker.placed_oranges))
                elif phase_before != "SPATIAL_RESET" and phase_after == "SPATIAL_RESET":
                    recorder.discard()
                elif phase_before != "ABORT_HOME" and phase_after == "ABORT_HOME":
                    recorder.discard()  # drop the failed subtask frames
                    recorder.start()    # arm fresh for the episode-ending home movement

                if phase_before != phase_after and phase_after in ("GRASP", "LIFT", "PLACE", "HOME"):
                    recorder.start()

            # --- Subtask success metrics (eval mode only) ---
            if not _fs_active:
                _np = len(sub_tracker.placed_oranges)
                if phase_before == "GRASP" and phase_after != "GRASP":
                    tracker.record_subtask_result("GRASP", _np, success=(phase_after == "LIFT"))
                elif phase_before == "LIFT" and phase_after != "LIFT":
                    tracker.record_subtask_result("LIFT", _np, success=(phase_after == "PLACE"))
                elif phase_before == "PLACE" and phase_after != "PLACE":
                    _ok = target_before in sub_tracker.placed_oranges
                    tracker.record_subtask_result("PLACE", _np - 1 if _ok else _np, success=_ok)

                tracker.record_timing(infer_time_ms, step_time_ms)
                tracker.update_step(step_count)

            is_terminated = tensor_to_bool(terminated)
            is_truncated  = tensor_to_bool(truncated)
            done = done or is_terminated or is_truncated

            if done:
                if recorder and not episode_succeeded:
                    recorder.discard()
                if not _fs_active:
                    final_positions = save_positions(env)
                    oranges_in_plate = orchestrated_oranges_in_plate(final_positions, sub_tracker)
                    story_success = oranges_in_plate >= len(controller.orange_names)
                    if story_success:
                        end_reason = "success_3_oranges"
                    elif is_terminated:
                        end_reason = "env_terminated"
                    elif is_truncated:
                        end_reason = "env_truncated"
                    else:
                        end_reason = "episode_finished"
                    story_record = finish_story_episode(
                        episode_story, step_count, oranges_in_plate, end_reason, story_success,
                        plate_pos, orange_positions, sub_tracker, controller
                    )
                    tracker.end_episode(run_idx, step_count, is_terminated, oranges_in_plate,
                                        n_local_retries=controller.n_local_retries,
                                        n_redirections=controller.n_redirections,
                                        n_oranges_abandoned=sum(1 for c in controller._timeout_count.values() if c >= 2),
                                        camera_images=last_camera_images,
                                        episode_story=story_record)

        # --- Post-episode: merge staging on full success ---
        if _fs_active and RECORD_ENABLED:
            recorder.close_writers()
            if episode_succeeded:
                n_merged = merge_staging_into(
                    SYNTHETIC_DATASETS_DIR / (RECORD_DATASET_NAME + "_staging"),
                    SYNTHETIC_DATASETS_DIR / RECORD_DATASET_NAME,
                )
                _fs_successes += 1
                print(f"\n  ✅ Full success {_fs_successes}/{n_inference_runs}"
                      f"  ({n_merged} subtasks merged, attempt {_fs_attempts})")
            else:
                print(f"\n  ❌ Attempt {_fs_attempts} failed — staging discarded")
            torch.cuda.empty_cache()

        if reset_controller.stop_requested:
            break

        if _fs_active and _fs_successes >= n_inference_runs:
            break

    upload_after_shutdown = True

except KeyboardInterrupt:
    print("\nInterrupted — saving evaluation summary and closing writers.")
except Exception as exc:
    print(f"\n❌ CRASH DETECTED: {exc}")
    import traceback
    traceback.print_exc()
finally:
    if _fs_active:
        print(f"\n  Full-success generation complete: {_fs_successes}/{n_inference_runs}"
              f" successes in {_fs_attempts} attempts.")
        # Per-episode staging recorders are already closed inside the loop.
        # Close any recorder that may still be open if we crashed mid-episode.
        try:
            if recorder:
                recorder.close_writers()
        except Exception:
            pass
    else:
        # Always print and save the evaluation summary, regardless of how the script exits.
        # (finally runs for normal completion, KeyboardInterrupt, and Python exceptions;
        # only os._exit()/SIGKILL can bypass this.)
        tracker.print_final_summary(model_id)
        if recorder:
            recorder.close_writers()
    print("Closing environment...")
    env.close()

# Push outside the Isaac Sim try/finally so Hub upload errors are visible
# and Isaac Sim shutdown logs don't bury them.
if RECORD_ENABLED and upload_after_shutdown:
    if _fs_active:
        # In full-success mode open the completed main dataset for Hub upload.
        print("\n📤 Pushing dataset to HuggingFace Hub...")
        _push_recorder = SubtaskRecorder.create(RECORD_DATASET_NAME, resume=True, freeze_frames=FREEZE_FRAMES)
        _push_recorder.push_to_hub()
    elif recorder:
        print("\n📤 Pushing dataset to HuggingFace Hub (this may take a few minutes for video data)...")
        recorder.push_to_hub()
