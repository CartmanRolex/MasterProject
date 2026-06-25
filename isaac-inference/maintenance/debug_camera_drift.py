"""
debug_camera_drift.py

Loads the pick-orange environment and performs rapid resets, logging the front
camera pose (POSITION *and* ORIENTATION) at every reset, distinguishing two
things that the old version conflated:

  REFERENCE - the baseline the randomization event adds its noise to, i.e.
              exactly what `randomize_camera_uniform` reads as `ori_pos_w` /
              `ori_quat_w` (`asset.data.pos_w` / `quat_w_*`). If this grows
              episode-over-episode, randomization is compounding -> drift.
  APPLIED   - the actual camera pose set into the simulation, read straight from
              the USD view (`asset._view.get_world_poses()`). This is ground
              truth, independent of the `data.pos_w` buffer.

Why hook the event instead of reading `data.pos_w` from outside (as before):
`data.pos_w` is only refreshed by `Camera._update_poses()`, at a time that
varies by machine (GPU / Isaac patch level / render timing). That staleness is
exactly what makes this bug look like a random walk on one machine and "frozen"
on another. Reading it from outside the event is therefore unreliable; wrapping
the event term lets us capture the true reference and the true applied pose on
every machine.

Run on both computers and compare camera_drift_log_<hostname>.txt.

Usage:
    ./remote.sh debug_camera_drift.py
"""

import math
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

# These need the SimulationApp running, so import after make_env.
from isaaclab.utils.math import (  # noqa: E402
    convert_camera_frame_orientation_convention,
    euler_xyz_from_quat,
)

# ── Hook the camera randomization event ───────────────────────────────────────
# The EventManager already holds a reference to the original event function in
# its term cfg, so we swap that term's `.func` with a wrapper. The wrapper records
# the reference pose (what randomization is added to) and the applied pose (USD
# ground truth) into `_capture`, which the main loop reads back and logs.

_capture: dict = {}


def _wrapped_randomize(env, env_ids, asset_cfg, pose_range, convention="ros", _orig=None):
    asset = env.scene[asset_cfg.name]

    # Fallback reference for the old event implementation, which used
    # asset.data as its baseline.
    fallback_ref_pos = asset.data.pos_w[0].clone()
    if convention == "ros":
        fallback_ref_quat = asset.data.quat_w_ros[0].clone()
    elif convention == "opengl":
        fallback_ref_quat = asset.data.quat_w_opengl[0].clone()
    else:
        fallback_ref_quat = asset.data.quat_w_world[0].clone()

    # Run the real randomization.
    _orig(env, env_ids, asset_cfg, pose_range, convention)

    cached_default_pose = getattr(asset, f"_leisaac_randomize_camera_uniform_defaults_{convention}", None)
    if cached_default_pose is None:
        ref_pos = fallback_ref_pos
        ref_quat = fallback_ref_quat
    else:
        ref_pos = cached_default_pose[0][0].clone()
        ref_quat = cached_default_pose[1][0].clone()

    # APPLIED: read the true pose from the USD view, NOT data.pos_w (which is
    # only refreshed by _update_poses at a machine-dependent time). The view
    # quaternion is in opengl convention; convert it to match `convention` so it
    # is directly comparable to the reference quaternion.
    view_pos, view_quat_opengl = asset._view.get_world_poses()
    applied_pos = view_pos[0].clone()
    applied_quat = convert_camera_frame_orientation_convention(
        view_quat_opengl, origin="opengl", target=convention
    )[0].clone()

    _capture["ref_pos"] = ref_pos
    _capture["ref_quat"] = ref_quat
    _capture["applied_pos"] = applied_pos
    _capture["applied_quat"] = applied_quat


def _hook_camera_event() -> None:
    for cfgs in env.event_manager._mode_term_cfgs.values():
        for cfg in cfgs:
            if getattr(cfg.func, "__name__", None) == "randomize_camera_uniform":
                orig = cfg.func
                cfg.func = lambda *a, _orig=orig, **kw: _wrapped_randomize(*a, _orig=_orig, **kw)
                return
    raise RuntimeError("Could not find a 'randomize_camera_uniform' event term to hook.")


_hook_camera_event()
print("Camera randomization event hooked.\n")

# ── Formatting helpers ────────────────────────────────────────────────────────

_log_lines: list[str] = []


def _pos_xyz(p) -> list[float]:
    return [float(v) for v in p.detach().cpu().flatten()[:3]]


def _rpy_deg(quat) -> list[float]:
    r, p, y = euler_xyz_from_quat(quat.detach().reshape(1, 4))
    return [math.degrees(float(r)), math.degrees(float(p)), math.degrees(float(y))]


def _fmt3(v) -> str:
    return f"[{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}]"


def _emit(line: str) -> None:
    print(line)
    _log_lines.append(line)


def log_episode(episode: int) -> None:
    if not _capture:
        _emit(f"ep={episode:04d}  (no camera randomization captured this reset)")
        return

    ref_p = _pos_xyz(_capture["ref_pos"])
    app_p = _pos_xyz(_capture["applied_pos"])
    ref_rpy = _rpy_deg(_capture["ref_quat"])
    app_rpy = _rpy_deg(_capture["applied_quat"])
    dpos = [a - r for a, r in zip(app_p, ref_p)]
    drpy = [a - r for a, r in zip(app_rpy, ref_rpy)]

    _emit(f"ep={episode:04d}  REFERENCE  pos_m={_fmt3(ref_p)}  rpy_deg={_fmt3(ref_rpy)}")
    _emit(f"ep={episode:04d}  APPLIED    pos_m={_fmt3(app_p)}  rpy_deg={_fmt3(app_rpy)}")
    _emit(f"ep={episode:04d}  DELTA      dpos_m={_fmt3(dpos)}  drpy_deg={_fmt3(drpy)}")


# Zero action (6-DOF SO-101).
_action_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 6
_zero_action = torch.zeros(1, _action_dim, device=env.device)

# ── Main loop ─────────────────────────────────────────────────────────────────

N_EPISODES = 200
N_STEPS = 1  # just enough to flush physics; the logged pose comes from the hook

print(f"Running {N_EPISODES} rapid resets...\n")
print("REFERENCE = baseline randomization is added to | APPLIED = true pose set in sim")
print("DELTA = APPLIED - REFERENCE (should stay within +/-0.025 m and +/-2.5 deg)")
print("Drift shows up as the REFERENCE values growing episode-over-episode.\n")
print("-" * 96)

for episode in range(N_EPISODES):
    _capture.clear()
    env.reset()  # triggers the hooked camera randomization
    for _ in range(N_STEPS):
        env.step(_zero_action)
    log_episode(episode)

print("-" * 96)

# ── Save log ──────────────────────────────────────────────────────────────────

host = socket.gethostname()
output_path = Path(__file__).parent / f"camera_drift_log_{host}.txt"

with open(output_path, "w") as f:
    f.write(f"host={host}  episodes={N_EPISODES}\n")
    f.write("REFERENCE = baseline randomization is added to | APPLIED = true pose set in sim\n")
    f.write("DELTA = APPLIED - REFERENCE | drift = REFERENCE growing episode-over-episode\n")
    f.write("=" * 96 + "\n")
    for line in _log_lines:
        f.write(line + "\n")

print(f"\nLog saved -> {output_path}")
print("Share this file for comparison.")
