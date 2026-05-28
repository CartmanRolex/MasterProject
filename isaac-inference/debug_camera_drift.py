"""
debug_camera_drift.py

Loads the pick-orange environment and performs rapid resets, logging the front
camera world position at every stage of the reset cycle.

Run on both computers and share camera_drift_log.txt to compare.

Usage:
    ./remote.sh debug_camera_drift.py
"""

import socket
from pathlib import Path

import torch

# ── Patch randomize_camera_uniform BEFORE make_env, so the EventTerm captures ──
# the patched version when PickOrangeEnvCfg.__post_init__ runs.

import leisaac.enhance.envs.mdp.events as _events_mod

_log_lines: list[str] = []
_episode: list[int] = [0]


def _fmt(pos) -> str:
    p = pos.cpu().numpy().flatten()
    return f"[{p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}]"


_original_fn = _events_mod.randomize_camera_uniform


def _patched_randomize_camera_uniform(env, env_ids, asset_cfg, pose_range, convention="ros"):
    from isaaclab.sensors import Camera
    asset: Camera = env.scene[asset_cfg.name]

    # This is exactly what the original reads as its base pose.
    # If it drifts from the default over episodes → accumulation bug confirmed.
    ori_pos_w = asset.data.pos_w[env_ids].clone()

    line = f"ep={_episode[0]:04d}  IN_RANDOMIZE  ori_pos_w={_fmt(ori_pos_w[0])}"
    print(line)
    _log_lines.append(line)

    _original_fn(env, env_ids, asset_cfg, pose_range, convention)


_events_mod.randomize_camera_uniform = _patched_randomize_camera_uniform

# ── Load environment ──────────────────────────────────────────────────────────

from lerobot.envs.factory import make_env  # noqa: E402  (after patch)

print("Loading LeIsaac pick-orange environment...")
envs_dict = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][0].envs[0].unwrapped

front_camera = env.scene["front"]
print("Environment ready.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def log_camera(label: str) -> None:
    pos = front_camera.data.pos_w[0]
    line = f"ep={_episode[0]:04d}  {label:<22s}  pos_w={_fmt(pos)}"
    print(line)
    _log_lines.append(line)


# Zero action (6-DOF SO-101). Use whatever shape the env expects.
_action_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 6
_zero_action = torch.zeros(1, _action_dim, device=env.device)

# ── Main loop ─────────────────────────────────────────────────────────────────

N_EPISODES = 200
N_STEPS = 3  # just enough to flush physics; keep it fast

print(f"Running {N_EPISODES} rapid resets ({N_STEPS} steps each)...\n")
print(f"{'ep':>4}  {'label':<22}  {'pos_w (x, y, z)'}")
print("-" * 72)

for episode in range(N_EPISODES):
    _episode[0] = episode

    # Camera data.pos_w here = pose read by camera.reset()/_update_poses()
    # at the START of this reset (reflects end-of-previous-episode state).
    # Then IN_RANDOMIZE will fire and show the same value as ori_pos_w.
    obs, _ = env.reset()
    log_camera("AFTER_RESET(data.pos_w)")

    # Take a few steps so the physics propagates and camera.update() runs,
    # giving a fresh data.pos_w that reflects the actual randomized pose.
    for step in range(N_STEPS):
        obs, _, _, _, _ = env.step(_zero_action)
        if step == 0:
            log_camera("AFTER_1ST_STEP(data.pos_w)")

print("-" * 72)

# ── Save log ──────────────────────────────────────────────────────────────────

host = socket.gethostname()
output_path = Path(__file__).parent / f"camera_drift_log_{host}.txt"

with open(output_path, "w") as f:
    f.write(f"host={host}  episodes={N_EPISODES}  steps_per_ep={N_STEPS}\n")
    f.write("=" * 72 + "\n")
    f.write(f"{'ep':>4}  {'label':<22}  {'pos_w (x, y, z)'}\n")
    f.write("-" * 72 + "\n")
    for line in _log_lines:
        f.write(line + "\n")

print(f"\nLog saved → {output_path}")
print("Share this file for comparison.")
