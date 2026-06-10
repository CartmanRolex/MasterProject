import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame
from tqdm import tqdm

from eval_utils import (
    camera_image_to_hwc_uint8,
    capture_initial_scene_audit,
    count_oranges_in_plate,
    evaluation_checkpoint_path,
    evaluation_summary_path,
    is_plate_upside_down,
    load_eval_seed_set,
    refresh_observation_after_reset,
    save_episode_camera_snapshots,
    save_positions,
    seed_metadata_for_episode,
    seed_set_checkpoint_metadata,
    short_model_name,
)
from robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)
from phase_monitor import PhaseMonitor

# ==========================================
# 1. Configuration & Setup
# ==========================================
def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "2"}


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_value(name, default=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = env_value("MODEL_ID", "MasterProject2026/Gal-merged-tailed-auto-no-lang-no-home")
n_inference_runs = env_int("N_INFERENCE_RUNS", 100)
max_steps = env_int("MAX_STEPS", 5000)
ACTIONS_PER_CHUNK = env_int("ACTIONS_PER_CHUNK", 20)
instruction = env_value("INSTRUCTION", "Grab orange and place into plate")

EVAL_RESUME = env_flag("EVAL_RESUME", True)
EVAL_RESULT_NAME = env_value("EVAL_RESULT_NAME")
EVAL_CHECKPOINT_PATH = env_value("EVAL_CHECKPOINT_PATH")
EVAL_SUMMARY_PATH = env_value("EVAL_SUMMARY_PATH")
EVAL_SEED_LIST_PATH = env_value("EVAL_SEED_LIST_PATH")
SAVE_CAMERA_SNAPSHOTS = env_flag("SAVE_CAMERA_SNAPSHOTS", True)
eval_seed_set = load_eval_seed_set(EVAL_SEED_LIST_PATH, min_count=n_inference_runs)


ENABLE_LIVESTREAM = env_flag("ENABLE_LIVESTREAM", env_flag("LIVESTREAM", False))

dataset_features = {
    "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["front"]},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["wrist"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (6,), "names": ["action"]},
}

KNOWN_BAD_DRIVER_PREFIXES = ("595.",)


def check_isaac_driver_compatibility():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return

    for line in output.splitlines():
        if not line.strip():
            continue
        name, _, driver = line.partition(",")
        gpu_name = name.strip()
        driver_version = driver.strip()
        is_blackwell_geforce = "RTX 50" in gpu_name or "RTX 5080" in gpu_name or "RTX 5090" in gpu_name
        if is_blackwell_geforce and driver_version.startswith(KNOWN_BAD_DRIVER_PREFIXES):
            print(
                "\nIsaac Sim is likely to crash before Python can handle it.\n"
                f"Detected GPU/driver: {gpu_name}, driver {driver_version}.\n"
                "Isaac Sim 5.1 has known RTX renderer startup crashes with 595.xx drivers on Blackwell GPUs.\n"
                "Install NVIDIA's Isaac Sim 5.1 validated Linux driver, 580.65.06, or another non-595 driver,\n"
                "then rerun this script.",
                file=sys.stderr,
            )
            sys.exit(1)


def tensor_to_bool(value):
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


class FlatEvaluationTracker:
    INFER_THRESHOLD_MS = 5.0

    def __init__(
        self,
        n_episodes,
        model_id,
        checkpoint_path=None,
        summary_path=None,
        resume=True,
        result_name=None,
        seed_set=None,
    ):
        self.n_episodes = n_episodes
        self.model_id = model_id
        self.result_name = result_name
        self.seed_set = seed_set
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self._default_checkpoint_path(model_id, result_name)
        self.summary_path = Path(summary_path) if summary_path else self._default_summary_path(model_id, result_name)
        self.episode_records = []
        self._infer_times = []
        self._step_times = []

        if resume:
            self._load_checkpoint()

        self._pbar = tqdm(
            total=n_episodes,
            initial=min(len(self.episode_records), n_episodes),
            desc="Episodes",
            unit="ep",
        )

    @staticmethod
    def _short_model_name(model_id):
        return short_model_name(model_id)

    @classmethod
    def _default_checkpoint_path(cls, model_id, result_name=None):
        return evaluation_checkpoint_path(model_id, flat=True, result_name=result_name)

    @classmethod
    def _default_summary_path(cls, model_id, result_name=None):
        return evaluation_summary_path(model_id, flat=True, result_name=result_name)

    @property
    def next_episode_index(self):
        if not self.episode_records:
            return 0
        return max(record["episode"] for record in self.episode_records) + 1

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return

        try:
            with open(self.checkpoint_path) as f:
                checkpoint = json.load(f)
        except Exception as exc:
            tqdm.write(f"  Could not read flat evaluation checkpoint {self.checkpoint_path}: {exc}")
            return

        checkpoint_model = checkpoint.get("model_id")
        if checkpoint_model and checkpoint_model != self.model_id:
            tqdm.write(
                f"  Ignoring flat evaluation checkpoint for {checkpoint_model}; "
                f"current model is {self.model_id}"
            )
            return

        records = checkpoint.get("episodes", [])
        self.episode_records = sorted(
            (record for record in records if isinstance(record.get("episode"), int)),
            key=lambda record: record["episode"],
        )
        if self.episode_records:
            tqdm.write(
                f"  Resuming flat evaluation: {len(self.episode_records)} "
                f"completed run(s) loaded from {self.checkpoint_path}"
            )

    def start_episode(self):
        self._infer_times = []
        self._step_times = []

    def record_timing(self, infer_time_ms, step_time_ms):
        if infer_time_ms is not None and infer_time_ms > self.INFER_THRESHOLD_MS:
            self._infer_times.append(infer_time_ms)
        if step_time_ms is not None:
            self._step_times.append(step_time_ms)

    def end_episode(
        self,
        episode,
        step_count,
        is_terminated,
        oranges_in_plate,
        camera_images=None,
        seed_metadata=None,
        initial_scene=None,
        start_camera_images=None,
        phase_debug=None,
    ):
        n_infer_calls = len(self._infer_times)
        last_infer = self._infer_times[-1] if self._infer_times else float("nan")
        avg_step = (sum(self._step_times) / len(self._step_times)) if self._step_times else float("nan")
        outcome = "TERMINATED" if is_terminated else "TRUNCATED"
        was_new_episode = all(record["episode"] != episode for record in self.episode_records)

        record = {
            "episode": int(episode),
            "step_count": int(step_count),
            "is_terminated": bool(is_terminated),
            "oranges_in_plate": int(oranges_in_plate),
            "n_infer_calls": int(n_infer_calls),
            "last_infer_ms": last_infer,
            "avg_step_ms": avg_step,
            "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if seed_metadata:
            record.update(seed_metadata)
        if initial_scene:
            record["initial_scene_audit"] = initial_scene
        if phase_debug:
            record.update(phase_debug)
        self.episode_records = [r for r in self.episode_records if r["episode"] != episode]
        self.episode_records.append(record)
        self.episode_records.sort(key=lambda r: r["episode"])

        tqdm.write(
            f"  Episode {episode:>3d} | {outcome:<10s} | "
            f"Oranges: {oranges_in_plate}/3 | "
            f"Steps: {step_count:>4d} | "
            f"Infer: {n_infer_calls} real calls, last {last_infer:.0f} ms | "
            f"Avg step: {avg_step:>6.1f} ms"
        )
        if was_new_episode:
            self._pbar.update(1)
        if SAVE_CAMERA_SNAPSHOTS and start_camera_images:
            try:
                save_episode_camera_snapshots(
                    self.model_id,
                    "flat",
                    episode,
                    start_camera_images.get("front"),
                    start_camera_images.get("wrist"),
                    result_name=self.result_name,
                    stage="start",
                )
            except Exception as exc:
                tqdm.write(f"  Could not save flat episode start camera snapshots: {exc}")
        if SAVE_CAMERA_SNAPSHOTS and camera_images:
            try:
                save_episode_camera_snapshots(
                    self.model_id,
                    "flat",
                    episode,
                    camera_images.get("front"),
                    camera_images.get("wrist"),
                    result_name=self.result_name,
                )
            except Exception as exc:
                tqdm.write(f"  Could not save flat episode camera snapshots: {exc}")
        self.save_checkpoint()
        self.write_summary()

    def _summary_text(self):
        records = self.episode_records
        n_eval = len(records)
        oranges = [record["oranges_in_plate"] for record in records]
        successes = sum(1 for count in oranges if count == 3)
        success_steps = [record["step_count"] for record in records if record["oranges_in_plate"] == 3]
        pct = lambda count: (count / n_eval * 100) if n_eval else 0
        avg_oranges = sum(oranges) / n_eval if n_eval else 0
        mean_success_steps = sum(success_steps) / len(success_steps) if success_steps else float("nan")
        header = (
            "FLAT-PROMPT EVALUATION COMPLETE"
            if n_eval == self.n_episodes
            else f"FLAT-PROMPT EVALUATION SUMMARY (stopped after {n_eval}/{self.n_episodes} runs)"
        )
        seed_set = seed_set_checkpoint_metadata(self.seed_set)
        seed_line = f"Seed set:             {seed_set['name']} ({seed_set['sha256']})\n" if seed_set else ""

        return (
            f"\n========================================\n"
            f"{header}\n"
            f"Model ID:             {self.model_id}\n"
            f"Result name:          {self.result_name or short_model_name(self.model_id)}\n"
            f"Prompt:               {instruction}\n"
            f"{seed_line}"
            f"Success Rate:         {successes}/{n_eval} ({pct(successes):.2f}%)\n"
            f"Avg oranges in plate: {avg_oranges:.2f}/3\n"
            f"Mean steps (success): {mean_success_steps:.1f}\n"
            f"3/3 oranges:          {oranges.count(3)}/{n_eval} ({pct(oranges.count(3)):.1f}%)\n"
            f"2/3 oranges:          {oranges.count(2)}/{n_eval} ({pct(oranges.count(2)):.1f}%)\n"
            f"1/3 oranges:          {oranges.count(1)}/{n_eval} ({pct(oranges.count(1)):.1f}%)\n"
            f"0/3 oranges:          {oranges.count(0)}/{n_eval} ({pct(oranges.count(0)):.1f}%)\n"
            f"Per-episode oranges:  {oranges}\n"
            f"========================================\n"
        )

    def save_checkpoint(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_id": self.model_id,
            "result_name": self.result_name,
            "prompt": instruction,
            "seed_set": seed_set_checkpoint_metadata(self.seed_set),
            "target_n_episodes": self.n_episodes,
            "completed_episodes": len(self.episode_records),
            "last_update": datetime.datetime.now().isoformat(timespec="seconds"),
            "episodes": self.episode_records,
        }
        tmp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(checkpoint, f, indent=2)
        tmp.replace(self.checkpoint_path)

    def write_summary(self):
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(self._summary_text())
        tmp.replace(self.summary_path)

    def print_final_summary(self):
        self._pbar.close()
        print(self._summary_text())
        self.write_summary()
        print(f"Summary saved to: {self.summary_path}\n")


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
            elif raw in ("s", "stop"):
                with self._lock:
                    self._stop_requested = True
                print("\nStop requested - will save after this episode.")


# ==========================================
# 2. Environment & Policy Initialization
# ==========================================
os.environ["ENABLE_CAMERAS"] = "1"
if ENABLE_LIVESTREAM:
    os.environ["LIVESTREAM"] = os.environ.get("LIVESTREAM", "2")
    os.environ["HEADLESS"] = "0"
else:
    os.environ["LIVESTREAM"] = "0"
    os.environ["HEADLESS"] = "1"
print(f"Livestream: {'enabled' if ENABLE_LIVESTREAM else 'disabled'} (LIVESTREAM={os.environ['LIVESTREAM']})")

check_isaac_driver_compatibility()

print("Loading LeIsaac Environment...")
envs_dict = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][0].envs[0].unwrapped
env.cfg.episode_length_s = max_steps * env.cfg.sim.dt * env.cfg.decimation

print(f"Loading trained policy: {model_id}...")
policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
model_chunk_size = getattr(policy.config, "chunk_size", None)
execute_actions_per_chunk = (
    min(ACTIONS_PER_CHUNK, model_chunk_size) if isinstance(model_chunk_size, int) else ACTIONS_PER_CHUNK
)
policy.config.n_action_steps = execute_actions_per_chunk
policy.reset()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

logging.getLogger("omni").setLevel(logging.ERROR)
logging.getLogger("carb").setLevel(logging.ERROR)
try:
    import carb

    carb.settings.get_settings().set_string("/log/level", "error")
except ImportError:
    pass


# ==========================================
# 3. Evaluation Loop
# ==========================================
tracker = FlatEvaluationTracker(
    n_inference_runs,
    model_id=model_id,
    checkpoint_path=EVAL_CHECKPOINT_PATH,
    summary_path=EVAL_SUMMARY_PATH,
    resume=EVAL_RESUME,
    result_name=EVAL_RESULT_NAME,
    seed_set=eval_seed_set,
)
reset_controller = ResetController()
reset_controller.start()


def _shutdown_handler(_sig, _frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)

print(f"""
{'=' * 52}
  FLAT-PROMPT EVALUATION
  Model:          {model_id}
  Result name:    {EVAL_RESULT_NAME or short_model_name(model_id)}
  Prompt:         "{instruction}"
  Model chunk:    {model_chunk_size if model_chunk_size is not None else 'unknown'} actions
  Execute chunk:  {execute_actions_per_chunk} actions before replanning
  Inference runs: {n_inference_runs}
  Max steps:      {max_steps}
  Seed set:       {eval_seed_set['name'] if eval_seed_set else 'unseeded'}
  Completed runs: {len(tracker.episode_records)} already tracked
{'=' * 52}
""")

try:
    for episode in range(tracker.next_episode_index, n_inference_runs):
        print(f"\n{'-' * 52}")
        print(f"  Flat-prompt run {episode + 1} / {n_inference_runs}")
        print(f"{'-' * 52}")
        print(f'  ACTIVE PROMPT -> "{instruction}"')

        episode_seed, seed_metadata = seed_metadata_for_episode(eval_seed_set, episode)
        if episode_seed is None:
            obs, _ = env.reset()
        else:
            print(f"  Seed: {episode_seed} (index {episode})")
            obs, _ = env.reset(seed=episode_seed)
        obs = refresh_observation_after_reset(env, obs)
        start_policy_obs = obs["policy"]
        start_camera_images = {
            "front": camera_image_to_hwc_uint8(start_policy_obs["front"][0].cpu().numpy()),
            "wrist": camera_image_to_hwc_uint8(start_policy_obs["wrist"][0].cpu().numpy()),
        }
        initial_scene = capture_initial_scene_audit(env)
        policy.reset()
        tracker.start_episode()
        phase_monitor = PhaseMonitor(model_id=model_id)
        phase_monitor.warm_up(env)

        done = False
        step_count = 0
        last_positions = save_positions(env)
        last_camera_images = None
        STABLE_PLATE_FRAMES = 10
        stable_plate_count = 0

        while not done:
            if reset_controller.stop_requested:
                print("\nStop requested - ending current episode as failed.")
                oranges_in_plate = count_oranges_in_plate(last_positions)
                phase_monitor.finish_line()
                tracker.end_episode(
                    episode,
                    step_count,
                    False,
                    oranges_in_plate,
                    camera_images=last_camera_images,
                    seed_metadata=seed_metadata,
                    initial_scene=initial_scene,
                    start_camera_images=start_camera_images,
                    phase_debug=phase_monitor.build_record(
                        episode,
                        step_count,
                        oranges_in_plate,
                        end_reason="stop_requested",
                        is_success=False,
                        final_positions=last_positions,
                    ),
                )
                done = True
                break

            if reset_controller.get_and_clear_reset():
                print("\nEpisode reset requested - ending current episode as failed.")
                oranges_in_plate = count_oranges_in_plate(last_positions)
                phase_monitor.finish_line()
                tracker.end_episode(
                    episode,
                    step_count,
                    False,
                    oranges_in_plate,
                    camera_images=last_camera_images,
                    seed_metadata=seed_metadata,
                    initial_scene=initial_scene,
                    start_camera_images=start_camera_images,
                    phase_debug=phase_monitor.build_record(
                        episode,
                        step_count,
                        oranges_in_plate,
                        end_reason="manual_reset",
                        is_success=False,
                        final_positions=last_positions,
                    ),
                )
                done = True
                break

            policy_obs = obs["policy"]
            raw_front = camera_image_to_hwc_uint8(policy_obs["front"][0].cpu().numpy())
            raw_wrist = camera_image_to_hwc_uint8(policy_obs["wrist"][0].cpu().numpy())
            last_camera_images = {"front": raw_front, "wrist": raw_wrist}
            joint_pos_converted = convert_leisaac_action_to_lerobot(policy_obs["joint_pos"].cpu().numpy())

            obs_frame = build_inference_frame(
                observation={
                    "front": raw_front,
                    "wrist": raw_wrist,
                    "state": joint_pos_converted[0],
                },
                ds_features=dataset_features,
                device=device,
                task=instruction,
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

            last_positions = save_positions(env)
            t_step_start = time.perf_counter()
            obs, _reward, terminated, truncated, _info = env.step(step_action[0].unsqueeze(0))
            step_time_ms = (time.perf_counter() - t_step_start) * 1000

            tracker.record_timing(infer_time_ms, step_time_ms)
            step_count += 1
            phase_monitor.update(env, step_count, episode)

            post_step_positions = save_positions(env)
            if is_plate_upside_down(post_step_positions):
                print("\nPlate flipped upside down - truncating episode as failure.")
                phase_monitor.finish_line()
                tracker.end_episode(
                    episode,
                    step_count,
                    False,
                    0,
                    camera_images=last_camera_images,
                    seed_metadata=seed_metadata,
                    initial_scene=initial_scene,
                    start_camera_images=start_camera_images,
                    phase_debug=phase_monitor.build_record(
                        episode,
                        step_count,
                        0,
                        end_reason="plate_flipped",
                        is_success=False,
                        final_positions=post_step_positions,
                    ),
                )
                done = True
                break

            if count_oranges_in_plate(post_step_positions) >= 3:
                stable_plate_count += 1
            else:
                stable_plate_count = 0

            is_terminated = tensor_to_bool(terminated) or stable_plate_count >= STABLE_PLATE_FRAMES
            is_truncated = tensor_to_bool(truncated)
            done = is_terminated or is_truncated

            if done:
                # Isaac Lab auto-resets during env.step() when truncated/terminated fires,
                # so post_step_positions reflects the reset state (count=0) in those cases.
                # Only use it when our own stability check triggered done; otherwise use
                # last_positions (captured before the resetting step), which matches the snapshot.
                if stable_plate_count >= STABLE_PLATE_FRAMES:
                    count_positions = post_step_positions
                else:
                    count_positions = last_positions
                oranges_in_plate = count_oranges_in_plate(count_positions)
                if stable_plate_count >= STABLE_PLATE_FRAMES:
                    end_reason = "all_oranges_stable_in_plate"
                elif is_truncated:
                    end_reason = "env_truncated"
                elif tensor_to_bool(terminated):
                    end_reason = "env_terminated"
                else:
                    end_reason = "episode_ended"
                phase_monitor.finish_line()
                tracker.end_episode(
                    episode,
                    step_count,
                    is_terminated,
                    oranges_in_plate,
                    camera_images=last_camera_images,
                    seed_metadata=seed_metadata,
                    initial_scene=initial_scene,
                    start_camera_images=start_camera_images,
                    phase_debug=phase_monitor.build_record(
                        episode,
                        step_count,
                        oranges_in_plate,
                        end_reason=end_reason,
                        is_success=oranges_in_plate >= 3,
                        final_positions=count_positions,
                    ),
                )

        if reset_controller.stop_requested:
            break

except KeyboardInterrupt:
    print("\nInterrupted - saving flat evaluation summary.")
except Exception as exc:
    print(f"\nCRASH DETECTED: {exc}")
    import traceback

    traceback.print_exc()
finally:
    tracker.print_final_summary()
    print("Closing environment...")
    env.close()
