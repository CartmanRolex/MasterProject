"""
Exploration branching script — determinism check edition.

Press 'c' + Enter during a run to:
  1. Checkpoint the current simulation state
  2. Run EXPLORE_STEPS of normal inference → save run1 video
  3. Rollback to checkpoint
  4. Run the exact same inference again (no changes) → save run2 video
  5. Rollback to checkpoint and continue the episode normally

If rollback is accurate, both videos should be pixel-identical.
Both videos land in explore_tests/ next to this file.
"""

import copy
import datetime
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
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


# ==========================================
# Configuration
# ==========================================
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id   = "MasterProject2026/Gal-pick-orange-tailedCH20"
n_episodes = 100
MAX_STEPS  = 5000
EXPLORE_STEPS = 200
EXPLORE_SEED  = 42
VIDEO_FPS     = 30
ORANGE_NAMES  = ("Orange001", "Orange002", "Orange003")

AVAILABLE_PROMPTS = [
    "Pick it up",
    "Place it into plate",
    "Go back to start position",
]

_EXPLORE_DIR = Path(__file__).parent / "explore_tests"

dataset_features = {
    "observation.images.front": {"dtype": "video", "shape": (3, 480, 640), "names": ["front"]},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["wrist"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (6,), "names": ["action"]},
}


# ==========================================
# SimCheckpoint
# ==========================================
@dataclass
class SimCheckpoint:
    obs: dict
    robot_joint_pos:  torch.Tensor
    robot_joint_vel:  torch.Tensor
    robot_root_state: torch.Tensor
    orange_states:    dict = field(default_factory=dict)
    plate_state:      torch.Tensor = None

    @staticmethod
    def save(env, obs: dict) -> "SimCheckpoint":
        robot = env.scene["robot"]
        orange_states = {
            name: env.scene[name].data.root_state_w[0].clone()
            for name in ORANGE_NAMES
        }
        return SimCheckpoint(
            obs=copy.deepcopy(obs),
            robot_joint_pos=robot.data.joint_pos[0].clone(),
            robot_joint_vel=robot.data.joint_vel[0].clone(),
            robot_root_state=robot.data.root_state_w[0].clone(),
            orange_states=orange_states,
            plate_state=env.scene["Plate"].data.root_state_w[0].clone(),
        )

    def restore(self, env) -> dict:
        robot = env.scene["robot"]
        robot.write_joint_position_to_sim(self.robot_joint_pos.unsqueeze(0))
        robot.write_joint_velocity_to_sim(self.robot_joint_vel.unsqueeze(0))
        robot.write_root_state_to_sim(self.robot_root_state.unsqueeze(0))
        for name, state in self.orange_states.items():
            env.scene[name].write_root_state_to_sim(state.unsqueeze(0))
        env.scene["Plate"].write_root_state_to_sim(self.plate_state.unsqueeze(0))
        env.scene.write_data_to_sim()
        return copy.deepcopy(self.obs)


# ==========================================
# Video helpers
# ==========================================
def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img


def _make_row(raw_front: np.ndarray, raw_wrist: np.ndarray) -> np.ndarray:
    """Return a 480×1280 RGB frame with front and wrist side by side."""
    return np.concatenate([_to_hwc_uint8(raw_front), _to_hwc_uint8(raw_wrist)], axis=1)


def _label_row(row: np.ndarray, text: str) -> np.ndarray:
    """Overlay a text label onto a 480×1280 RGB frame (in-place copy)."""
    out = row.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _write_combined_video(path: Path, frame_lists: list[list], labels: list[str], fps: int):
    """Write a single video with one row per run, stacked vertically, with labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(frame_lists)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (1280, 480 * n_rows))
    n_frames = max(len(f) for f in frame_lists)
    blank = np.zeros((480, 1280, 3), dtype=np.uint8)
    for i in range(n_frames):
        rows = [
            _label_row(frames[i] if i < len(frames) else blank, label)
            for frames, label in zip(frame_lists, labels)
        ]
        writer.write(cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    writer.release()


# ==========================================
# ExploreController
# ==========================================
class ExploreController:
    """Background thread that listens for terminal input.

    Commands:
      Enter alone  — print menu
      c            — trigger determinism branch at next step
      r / reset    — force-reset current episode
      t <N>        — set episode truncation to N steps
      n <N>        — set explore branch length
      0..K         — switch to numbered prompt
      any text     — set as custom prompt
    """

    def __init__(self, initial_prompt: str, prompts: list[str], env, explore_steps: int):
        self._lock              = threading.Lock()
        self._prompt            = initial_prompt
        self._prompts           = prompts
        self._env               = env
        self._reset_requested   = False
        self._explore_requested = False
        self._explore_steps     = explore_steps
        self._thread            = threading.Thread(target=self._listen, daemon=True)

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

    def get_and_clear_explore(self) -> bool:
        with self._lock:
            flag = self._explore_requested
            self._explore_requested = False
            return flag

    def explore_steps(self) -> int:
        with self._lock:
            return self._explore_steps

    def _current_max_steps(self) -> int:
        return self._env.max_episode_length

    def _set_max_steps(self, n: int):
        step_dt = self._env.cfg.sim.dt * self._env.cfg.decimation
        self._env.cfg.episode_length_s = n * step_dt

    def _print_menu(self):
        with self._lock:
            steps = self._explore_steps
        print("\n" + "=" * 60)
        print("  DETERMINISM TESTER")
        print("=" * 60)
        for i, p in enumerate(self._prompts):
            marker = "▶" if p == self._prompt else " "
            print(f"  {marker} [{i}] {p}")
        print(f"\n  Truncation:    {self._current_max_steps()} steps  (change: t <N>)")
        print(f"  Branch length: {steps} steps  (change: n <N>)")
        print("\n  c  → checkpoint + run two identical branches")
        print("  r  → reset episode")
        print("=" * 60 + "\n")

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
                print("\n  Episode reset requested.\n")
                continue

            if raw == "c":
                with self._lock:
                    self._explore_requested = True
                print("\n  Determinism branch requested — fires at next step.\n")
                continue

            if raw.startswith("t "):
                parts = raw.split()
                if len(parts) == 2 and parts[1].isdigit():
                    self._set_max_steps(int(parts[1]))
                    print(f"\n  Truncation set to {parts[1]} steps.\n")
                else:
                    print("\n  Usage: t <N>\n")
                continue

            if raw.startswith("n "):
                parts = raw.split()
                if len(parts) == 2 and parts[1].isdigit():
                    with self._lock:
                        self._explore_steps = int(parts[1])
                    print(f"\n  Branch length set to {parts[1]} steps.\n")
                else:
                    print("\n  Usage: n <N>\n")
                continue

            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(self._prompts):
                    with self._lock:
                        self._prompt = self._prompts[idx]
                    print(f"\n  Prompt → \"{self._prompts[idx]}\"\n")
                else:
                    print(f"\n  Index out of range. Choose 0–{len(self._prompts) - 1}.\n")
            else:
                with self._lock:
                    self._prompt = raw
                print(f"\n  Custom prompt: \"{raw}\"\n")


# ==========================================
# Policy step helper
# ==========================================
def policy_step(env, policy, preprocess, postprocess, obs, instruction):
    """Run one inference step; return (action_tensor_for_env, raw_front, raw_wrist, action_np)."""
    policy_obs = obs["policy"]
    raw_front  = policy_obs["front"][0].cpu().numpy()
    raw_wrist  = policy_obs["wrist"][0].cpu().numpy()

    joint_pos_converted = convert_leisaac_action_to_lerobot(
        policy_obs["joint_pos"].cpu().numpy()
    )
    obs_frame = build_inference_frame(
        observation={"front": raw_front, "wrist": raw_wrist, "state": joint_pos_converted[0]},
        ds_features=dataset_features,
        device=device,
        task=instruction,
    )
    batch = preprocess(obs_frame)
    with torch.inference_mode():
        action_output = policy.select_action(batch)

    action_dict  = postprocess(action_output)
    final_action = action_dict.get("action", action_dict) if isinstance(action_dict, dict) else action_dict
    action_np    = final_action.cpu().numpy()
    if action_np.ndim == 1:
        action_np = action_np[None, :]
    step_action = torch.from_numpy(convert_lerobot_action_to_leisaac(action_np)).to(device)
    return step_action[0].unsqueeze(0), raw_front, raw_wrist, action_np


def read_frames(obs):
    """Extract raw front/wrist images from obs without running the model."""
    policy_obs = obs["policy"]
    return policy_obs["front"][0].cpu().numpy(), policy_obs["wrist"][0].cpu().numpy()


# ==========================================
# Determinism branch runner
# ==========================================
def run_explore_branch(env, policy, preprocess, postprocess, obs, controller, step_count):
    n_steps     = controller.explore_steps()
    instruction = controller.get()
    ts          = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\n{'─'*60}")
    print(f"  DETERMINISM BRANCH  step={step_count}  n={n_steps}")
    print(f"  Checkpointing...")
    ckpt = SimCheckpoint.save(env, obs)

    # Save RNG state so normal inference is unaffected by branch seeding
    cpu_rng  = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # ── Branch 1: model reference, record actions ────────────────
    print(f"  Run 1: model reference ({n_steps} steps)...")
    obs_b1 = ckpt.restore(env)
    policy.reset()
    torch.manual_seed(EXPLORE_SEED)
    torch.cuda.manual_seed_all(EXPLORE_SEED)
    frames1, recorded_actions = [], []
    for _ in range(n_steps):
        step_action, rf, rw, _ = policy_step(env, policy, preprocess, postprocess, obs_b1, instruction)
        frames1.append(_make_row(rf, rw))
        recorded_actions.append(step_action.clone())
        obs_b1, _, terminated, truncated, _ = env.step(step_action)
        if terminated.any() or truncated.any():
            break

    # ── Branch 2: action replay, no model (physics check) ────────
    print(f"  Run 2: action replay ({len(recorded_actions)} steps)...")
    obs_b2 = ckpt.restore(env)
    frames2 = []
    for step_action in recorded_actions:
        rf, rw = read_frames(obs_b2)
        frames2.append(_make_row(rf, rw))
        obs_b2, _, terminated, truncated, _ = env.step(step_action)
        if terminated.any() or truncated.any():
            break

    # ── Branch 3: model re-run, same seed (model determinism check)
    print(f"  Run 3: model re-run ({n_steps} steps)...")
    obs_b3 = ckpt.restore(env)
    policy.reset()
    torch.manual_seed(EXPLORE_SEED)
    torch.cuda.manual_seed_all(EXPLORE_SEED)
    frames3 = []
    for _ in range(n_steps):
        step_action, rf, rw, _ = policy_step(env, policy, preprocess, postprocess, obs_b3, instruction)
        frames3.append(_make_row(rf, rw))
        obs_b3, _, terminated, truncated, _ = env.step(step_action)
        if terminated.any() or truncated.any():
            break

    out_path = _EXPLORE_DIR / f"explore_{ts}_step{step_count}.mp4"
    print(f"  Writing combined video...")
    _write_combined_video(
        out_path,
        frame_lists=[frames1, frames2, frames3],
        labels=[
            "Run 1: Model (reference)",
            "Run 2: Action replay (physics check)",
            "Run 3: Model re-run (determinism check)",
        ],
        fps=VIDEO_FPS,
    )

    # Restore RNG state so normal inference is not affected by branch seeding
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    restored_obs = ckpt.restore(env)
    policy.reset()

    print(f"  Saved: {out_path}  (1280x1440, three rows)")
    print(f"  Row 1 == Row 2 → physics is deterministic")
    print(f"  Row 1 == Row 3 → model+RNG is deterministic")
    print(f"{'─'*60}\n")

    return restored_obs


# ==========================================
# Environment & policy setup
# ==========================================
print("Loading LeIsaac Environment...")
envs_dict  = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)
suite_name = next(iter(envs_dict))
env        = envs_dict[suite_name][0].envs[0].unwrapped
env.cfg.episode_length_s = MAX_STEPS * env.cfg.sim.dt * env.cfg.decimation

print(f"Loading policy: {model_id}...")
policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

controller = ExploreController(
    initial_prompt=AVAILABLE_PROMPTS[0],
    prompts=AVAILABLE_PROMPTS,
    env=env,
    explore_steps=EXPLORE_STEPS,
)
controller.start()

import logging
logging.getLogger("omni").setLevel(logging.ERROR)
logging.getLogger("carb").setLevel(logging.ERROR)
try:
    import carb
    carb.settings.get_settings().set_string("/log/level", "error")
except ImportError:
    pass

print(f"\n--- STARTING: {n_episodes} EPISODES ---")


# ==========================================
# Main inference loop
# ==========================================
try:
    for episode in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()
        done       = False
        step_count = 0

        while not done:
            instruction = controller.get()
            step_action, _, _, _ = policy_step(env, policy, preprocess, postprocess, obs, instruction)
            obs, reward, terminated, truncated, info = env.step(step_action)
            step_count += 1

            if controller.get_and_clear_explore():
                obs = run_explore_branch(
                    env, policy, preprocess, postprocess, obs, controller, step_count
                )

            is_terminated = bool(terminated.item() if isinstance(terminated, torch.Tensor) else terminated)
            is_truncated  = bool(truncated.item()  if isinstance(truncated,  torch.Tensor) else truncated)
            done = is_terminated or is_truncated or controller.get_and_clear_reset()

        print(f"Episode {episode + 1} done after {step_count} steps.")

except KeyboardInterrupt:
    print("\nForce quitting...")
except Exception as e:
    print(f"\nCRASH: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Closing environment...")
    env.close()
