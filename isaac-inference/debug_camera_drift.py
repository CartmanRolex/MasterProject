"""
debug_camera_drift.py

Loads the pick-orange environment and performs rapid resets, logging the front
camera world position at every stage of the reset cycle.

Run on both computers and share camera_drift_log_<hostname>.txt to compare.

Usage:
    ./remote.sh debug_camera_drift.py
"""

import socket
from pathlib import Path

import torch

# SimulationApp must start before any isaaclab/leisaac imports.
# make_env handles that internally.
from lerobot.envs.factory import make_env

print("Loading LeIsaac pick-orange environment...")
envs_dict = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][0].envs[0].unwrapped

front_camera = env.scene["front"]
print("Environment ready.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

_log_lines: list[str] = []


def _fmt(pos) -> str:
    p = pos.cpu().numpy().flatten()
    return f"[{p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}]"


def log_camera(episode: int, label: str) -> None:
    pos = front_camera.data.pos_w[0]
    line = f"ep={episode:04d}  {label:<28s}  pos_w={_fmt(pos)}"
    print(line)
    _log_lines.append(line)


# Zero action (6-DOF SO-101).
_action_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 6
_zero_action = torch.zeros(1, _action_dim, device=env.device)

# ── Main loop ─────────────────────────────────────────────────────────────────

N_EPISODES = 200
N_STEPS = 3  # just enough to flush physics; keep it fast

print(f"Running {N_EPISODES} rapid resets ({N_STEPS} steps each)...\n")
print(f"{'ep':>4}  {'label':<28}  {'pos_w (x, y, z)'}")
print("-" * 80)

for episode in range(N_EPISODES):
    # Read BEFORE reset: this is what randomize_camera_uniform will see as
    # ori_pos_w (camera.reset() inside env.reset() calls _update_poses() which
    # reads the same USD stage value, so data.pos_w here == ori_pos_w inside
    # the event). If drift is accumulating, this value will grow each episode.
    log_camera(episode, "BEFORE_RESET(ori_pos_w_equiv)")

    obs, _ = env.reset()

    # After reset + physics flush: reflects the newly randomized camera pose.
    for step in range(N_STEPS):
        obs, _, _, _, _ = env.step(_zero_action)
        if step == 0:
            log_camera(episode, "AFTER_1ST_STEP(randomized)")

print("-" * 80)

# ── Save log ──────────────────────────────────────────────────────────────────

host = socket.gethostname()
output_path = Path(__file__).parent / f"camera_drift_log_{host}.txt"

with open(output_path, "w") as f:
    f.write(f"host={host}  episodes={N_EPISODES}  steps_per_ep={N_STEPS}\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'ep':>4}  {'label':<28}  {'pos_w (x, y, z)'}\n")
    f.write("-" * 80 + "\n")
    for line in _log_lines:
        f.write(line + "\n")

print(f"\nLog saved → {output_path}")
print("Share this file for comparison.")
