"""
Evaluation utilities for robotic pick-and-place tasks.

Provides:
  - save_positions / count_oranges_in_plate : geometric scene-state helpers
  - save_camera_snapshots                   : debug image dumps
  - EvaluationTracker                       : per-episode logging, tqdm bar, final summary + file export
  - SubtaskTracker                          : fine-grained phase tracker (multi-object pick-and-place)
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

# Always write results next to this file, regardless of working directory.
_RESULTS_DIR = Path(__file__).parent / "results"

# Rest-pose range for SO101 (degrees). Mirrors SO101_FOLLOWER_REST_POSE_RANGE in leisaac.
_SO101_REST_POSE_DEG = {
    "shoulder_pan":  (-30.0,   30.0),   # 0°
    "shoulder_lift": (-130.0, -70.0),   # -100°
    "elbow_flex":    (  60.0, 120.0),   # 90°
    "wrist_flex":    (  20.0,  80.0),   # 50°
    "wrist_roll":    ( -30.0,  30.0),   # 0°
    "gripper":       ( -40.0,  20.0),   # -10°
}

# Plate interior geometry. The interior is a truncated cone from
# PLATE_INNER_RADIUS (at PLATE_Z_MIN) widening to PLATE_RADIUS at
# PLATE_Z_MIN + PLATE_CONE_HEIGHT, then a straight cylinder of PLATE_RADIUS
# up to PLATE_Z_MAX.
PLATE_RADIUS       = 0.10    # XY radius of the cylindrical upper section
PLATE_INNER_RADIUS = 0.06    # XY radius at the bottom of the plate interior
PLATE_CONE_HEIGHT  = 0.035    # height of the truncated-cone section above PLATE_Z_MIN
PLATE_Z_MIN        = 0
PLATE_Z_MAX        = 0.065
PLATE_UPSIDE_DOWN_Z_THRESHOLD = 0.0


def plate_radius_at_height(z_offset,
                           plate_radius=PLATE_RADIUS,
                           inner_radius=PLATE_INNER_RADIUS,
                           cone_height=PLATE_CONE_HEIGHT,
                           plate_z_min=PLATE_Z_MIN):
    """Allowed XY radius of the plate interior at a height above the plate centre.

    Linearly interpolates from inner_radius at plate_z_min up to plate_radius at
    plate_z_min + cone_height (the truncated cone), then stays at plate_radius
    (the cylinder).
    """
    if cone_height <= 0:
        return plate_radius
    t = (z_offset - plate_z_min) / cone_height
    if t <= 0:
        return inner_radius
    if t >= 1:
        return plate_radius
    return inner_radius + t * (plate_radius - inner_radius)

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


# ============================================================
# Result Paths
# ============================================================

def short_model_name(model_id):
    return model_id.rstrip("/").split("/")[-1] if model_id else None


def evaluation_result_dir(model_id, result_name=None):
    model_name = short_model_name(result_name) or short_model_name(model_id)
    return _RESULTS_DIR / model_name if model_name else _RESULTS_DIR


def evaluation_checkpoint_path(model_id, flat=False, result_name=None):
    filename = "flat_checkpoint.json" if flat else "checkpoint.json"
    return evaluation_result_dir(model_id, result_name=result_name) / filename


def evaluation_summary_path(model_id, flat=False, result_name=None):
    filename = "flat_latest.txt" if flat else "latest.txt"
    return evaluation_result_dir(model_id, result_name=result_name) / filename


def evaluation_snapshot_dir(model_id, run_type, result_name=None):
    return evaluation_result_dir(model_id, result_name=result_name) / "snapshots" / run_type


def seed_list_sha256(seeds):
    """Hash the canonical seed array so all model runs can prove same episodes."""
    payload = json.dumps([int(seed) for seed in seeds], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_eval_seed_set(path, *, min_count=None):
    """Load and validate a reference seed JSON file.

    The expected format is a dict with a ``seeds`` list and optional metadata.
    A raw JSON list is accepted for quick experiments, but tracked reference
    files should use the dict form.
    """
    if not path:
        return None

    seed_path = Path(path).expanduser()
    if not seed_path.is_absolute():
        seed_path = Path(__file__).parent / seed_path
    data = json.loads(seed_path.read_text())
    if isinstance(data, list):
        data = {"name": seed_path.stem, "seeds": data}
    if not isinstance(data, dict) or not isinstance(data.get("seeds"), list):
        raise ValueError(f"seed file must be a JSON object with a seeds list: {seed_path}")

    seeds = [int(seed) for seed in data["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seed file contains duplicate seeds: {seed_path}")
    if min_count is not None and len(seeds) < int(min_count):
        raise ValueError(f"seed file has {len(seeds)} seeds, need at least {min_count}: {seed_path}")

    digest = seed_list_sha256(seeds)
    expected = data.get("sha256")
    if expected and str(expected) != digest:
        raise ValueError(f"seed file hash mismatch for {seed_path}: expected {expected}, got {digest}")
    count = data.get("count")
    if count is not None and int(count) != len(seeds):
        raise ValueError(f"seed file count mismatch for {seed_path}: declared {count}, got {len(seeds)}")

    return {
        "name": data.get("name") or seed_path.stem,
        "path": str(seed_path),
        "master_seed": data.get("master_seed"),
        "count": len(seeds),
        "seeds": seeds,
        "sha256": digest,
    }


def seed_metadata_for_episode(seed_set, episode):
    if not seed_set:
        return None, {}
    episode = int(episode)
    if episode < 0 or episode >= len(seed_set["seeds"]):
        raise IndexError(f"episode {episode} is outside seed set of length {len(seed_set['seeds'])}")
    seed = int(seed_set["seeds"][episode])
    return seed, {
        "seed": seed,
        "seed_index": episode,
        "seed_set_name": seed_set["name"],
        "seed_set_hash": seed_set["sha256"],
    }


def seed_set_checkpoint_metadata(seed_set):
    if not seed_set:
        return None
    return {
        "name": seed_set["name"],
        "path": seed_set["path"],
        "master_seed": seed_set.get("master_seed"),
        "count": seed_set["count"],
        "sha256": seed_set["sha256"],
    }


# ============================================================
# Camera Debugging
# ============================================================

def _camera_array_for_image(img):
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected image with 2 or 3 dimensions, got shape {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    arr = np.nan_to_num(arr)
    if arr.dtype != np.uint8:
        max_val = float(np.max(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def camera_image_to_hwc_uint8(img):
    """Return a camera image as HWC uint8 for LeRobot inference helpers."""
    return _camera_array_for_image(img)


def refresh_observation_after_reset(env, obs=None):
    """Advance one hold step after reset so sensor and rigid-body buffers are fresh.

    LeIsaac can return the previous episode's rendered camera tensors and COM
    buffers immediately after ``env.reset(seed=...)``. A single hold step updates
    the observation used for start snapshots and the first policy inference.
    """
    try:
        hold_action = env.scene["robot"].data.joint_pos[0].detach().clone()
    except Exception:
        if obs is None:
            raise
        hold_action = obs["policy"]["joint_pos"][0].detach().clone()

    if hold_action.ndim == 1:
        hold_action = hold_action.unsqueeze(0)
    refreshed_obs, _reward, _terminated, _truncated, _info = env.step(hold_action)
    return refreshed_obs


def save_episode_camera_snapshots(
    model_id,
    run_type,
    episode,
    raw_front,
    raw_wrist,
    *,
    result_name=None,
    stage="final",
):
    """Save final front/wrist images for an evaluated episode."""
    if raw_front is None or raw_wrist is None:
        return

    snapshot_dir = evaluation_snapshot_dir(model_id, run_type, result_name=result_name)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stage_suffix = "" if stage in (None, "", "final") else f"_{stage}"
    for name, img in (("front", raw_front), ("wrist", raw_wrist)):
        arr = _camera_array_for_image(img)
        Image.fromarray(arr).save(snapshot_dir / f"episode_{int(episode):06d}{stage_suffix}_{name}.png")

def save_camera_snapshots(raw_front, raw_wrist, episode, step_count,
                          target_step=15, target_episode=0):
    """Save front/wrist camera images at a specific (episode, step) for visual sanity-checking."""
    if episode != target_episode or step_count != target_step:
        return

    for name, img in [("front", raw_front), ("wrist", raw_wrist)]:
        Image.fromarray(_camera_array_for_image(img)).save(Path(__file__).parent / f"camera_check_{name}.png")
        print(f"  Saved camera_check_{name}.png")


# ============================================================
# Orange Position Classifier
# ============================================================

ORANGE_REF_AXIS          = "x"    # primary axis for left/right: "x" or "y"
ORANGE_INVERT            = True   # flip left/right on the primary axis
ORANGE_SECONDARY_INVERT  = True  # flip bottom/top on the secondary axis
ORANGE_AXIS_TOL          = 0.03    # oranges within this distance (m) on primary axis use secondary axis for disambiguation


def classify_orange_positions(orange_positions: dict) -> dict:
    """Return a label (e.g. 'left', 'middle', 'right', 'bottom left') for each orange.

    Sorts oranges along ORANGE_REF_AXIS to assign left/middle/right.
    If two adjacent oranges are within ORANGE_AXIS_TOL on the primary axis they are
    indistinguishable laterally, so they are split by the secondary horizontal axis
    (y if primary is x, x if primary is y) using bottom/top labels.
    """
    if len(orange_positions) != 3:
        return {name: name for name in orange_positions}

    primary_idx   = 0 if ORANGE_REF_AXIS == "x" else 1
    secondary_idx = 1 if ORANGE_REF_AXIS == "x" else 0

    vals  = {n: pos[primary_idx].item()   for n, pos in orange_positions.items()}
    sec   = {n: pos[secondary_idx].item() for n, pos in orange_positions.items()}

    sorted_names = sorted(vals, key=lambda n: vals[n])
    if ORANGE_INVERT:
        sorted_names = sorted_names[::-1]

    n0, n1, n2 = sorted_names
    close_01 = abs(vals[n0] - vals[n1]) < ORANGE_AXIS_TOL
    close_12 = abs(vals[n1] - vals[n2]) < ORANGE_AXIS_TOL

    def split_pair(a, b):
        """Return (bottom, top) names for a close pair using the secondary axis."""
        lo, hi = (a, b) if sec[a] <= sec[b] else (b, a)
        if ORANGE_SECONDARY_INVERT:
            lo, hi = hi, lo
        return lo, hi   # lo = "bottom", hi = "top"

    if close_01 and close_12:
        by_sec = sorted([n0, n1, n2], key=lambda n: sec[n])
        if ORANGE_SECONDARY_INVERT:
            by_sec = by_sec[::-1]
        sec_lbl = {by_sec[0]: "bottom", by_sec[1]: "middle", by_sec[2]: "top"}
        pri_lbl = {n0: "left", n1: "middle", n2: "right"}
        result  = {}
        for name in (n0, n1, n2):
            s, p = sec_lbl[name], pri_lbl[name]
            if s == "middle" and p == "middle":
                result[name] = "middle"
            elif s == "middle":
                result[name] = p               # "left" or "right"
            elif p == "middle":
                result[name] = s               # "bottom" or "top"
            else:
                result[name] = f"{s} {p}"     # "bottom left", "top right", etc.
        return result
    elif close_01:
        bot, top = split_pair(n0, n1)
        return {bot: "bottom left", top: "top left", n2: "right"}
    elif close_12:
        bot, top = split_pair(n1, n2)
        return {n0: "left", bot: "bottom right", top: "top right"}
    else:
        return {n0: "left", n1: "middle", n2: "right"}


# ============================================================
# Scene-State Helpers
# ============================================================

_MATH_UTILS = None


def _math_utils():
    """Lazily import isaaclab.utils.math.

    eval_utils is imported before the Isaac SimulationApp is launched, and
    importing isaaclab eagerly requires a running app. Deferring the import to
    first use (same pattern as inference_privileged_grasp.py) keeps module import
    cheap and app-independent.
    """
    global _MATH_UTILS
    if _MATH_UTILS is None:
        import isaaclab.utils.math as math_utils
        _MATH_UTILS = math_utils
    return _MATH_UTILS


def _position_component(pos, idx):
    value = pos[idx]
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return float(value)


def plate_up_vector_z(plate_quat):
    """Return the world-Z component of the plate local +Z axis."""
    if plate_quat is None:
        return 1.0

    quat = torch.as_tensor(plate_quat, dtype=torch.float32).flatten()
    if quat.numel() != 4:
        return 1.0

    norm = torch.linalg.vector_norm(quat)
    if norm.item() <= 0:
        return 1.0

    quat = quat / norm
    _w, x, y, _z = quat
    return (1.0 - 2.0 * (x * x + y * y)).item()


def is_plate_upside_down(positions_or_quat,
                         z_threshold=PLATE_UPSIDE_DOWN_Z_THRESHOLD):
    """Return True when the plate's local +Z axis points below world-horizontal."""
    if isinstance(positions_or_quat, dict):
        plate_quat = positions_or_quat.get("plate_quat")
    else:
        plate_quat = positions_or_quat
    return plate_up_vector_z(plate_quat) < z_threshold


def plate_position_metrics(plate_pos, orange_pos,
                           plate_radius=PLATE_RADIUS,
                           plate_z_min=PLATE_Z_MIN,
                           plate_z_max=PLATE_Z_MAX,
                           plate_quat=None,
                           inner_radius=PLATE_INNER_RADIUS,
                           cone_height=PLATE_CONE_HEIGHT):
    """Return shared plate-occupancy geometry for an orange COM position.

    The interior is a truncated cone (inner_radius → plate_radius over
    cone_height) topped by a straight cylinder, so the allowed XY radius grows
    with height near the bottom. When plate_quat (w,x,y,z) is given, the orange
    is transformed into the plate's local frame so the volume follows plate
    tilt. When None, the legacy world-axis-aligned volume is used.
    """
    if plate_quat is not None:
        plate_pos  = torch.as_tensor(plate_pos)
        orange_pos = torch.as_tensor(orange_pos, dtype=plate_pos.dtype, device=plate_pos.device)
        plate_quat = torch.as_tensor(plate_quat, dtype=plate_pos.dtype, device=plate_pos.device)
        local = _math_utils().quat_apply_inverse(plate_quat, orange_pos - plate_pos)
        xy_dist  = (local[0] ** 2 + local[1] ** 2).item() ** 0.5
        z_offset = local[2].item()
    else:
        px, py, pz = (_position_component(plate_pos, i) for i in range(3))
        ox, oy, oz = (_position_component(orange_pos, i) for i in range(3))
        xy_dist = ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5
        z_offset = oz - pz
    allowed_radius = plate_radius_at_height(
        z_offset, plate_radius, inner_radius, cone_height, plate_z_min
    )
    in_plate = xy_dist < allowed_radius and plate_z_min < z_offset < plate_z_max
    return xy_dist, z_offset, in_plate


def is_orange_position_in_plate(plate_pos, orange_pos,
                                plate_radius=PLATE_RADIUS,
                                plate_z_min=PLATE_Z_MIN,
                                plate_z_max=PLATE_Z_MAX,
                                plate_quat=None,
                                inner_radius=PLATE_INNER_RADIUS,
                                cone_height=PLATE_CONE_HEIGHT):
    """Return True when an orange COM is inside the shared plate volume."""
    _, _, in_plate = plate_position_metrics(
        plate_pos,
        orange_pos,
        plate_radius=plate_radius,
        plate_z_min=plate_z_min,
        plate_z_max=plate_z_max,
        plate_quat=plate_quat,
        inner_radius=inner_radius,
        cone_height=cone_height,
    )
    return in_plate


def save_positions(env, plate_name="Plate",
                   orange_names=("Orange001", "Orange002", "Orange003")):
    """Snapshot plate root position and orange COM positions, relative to env origin.

    Should be called every step. The last snapshot before done=True reflects the
    true final state before the env auto-resets.

    Returns:
        dict mapping "plate" to its root position, "plate_quat" to the plate's
        orientation quaternion (w,x,y,z), and each orange name to its
        centre-of-mass position.
    """
    origin = env.scene.env_origins[0]
    positions = {"plate": env.scene[plate_name].data.root_pos_w[0].clone() - origin}
    positions["plate_quat"] = env.scene[plate_name].data.root_quat_w[0].clone()
    for name in orange_names:
        positions[name] = env.scene[name].data.root_com_pos_w[0].clone() - origin
    return positions


def _serializable_vec(value, ndigits=8):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    elif isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    return [round(float(v), ndigits) for v in value]


def capture_initial_scene_audit(
    env,
    *,
    plate_name="Plate",
    orange_names=("Orange001", "Orange002", "Orange003"),
    camera_names=("front", "wrist"),
):
    """Capture reset-time object poses and camera USD view poses for seed audits."""
    origin = env.scene.env_origins[0]
    plate = env.scene[plate_name]
    scene = {
        "relative_to_env_origin": True,
        "plate": {
            "name": plate_name,
            "position": _serializable_vec(plate.data.root_pos_w[0] - origin),
            "quat_wxyz": _serializable_vec(plate.data.root_quat_w[0]),
        },
        "oranges": {},
        "cameras": {},
    }

    for name in orange_names:
        orange = env.scene[name]
        scene["oranges"][name] = {
            "com_position": _serializable_vec(orange.data.root_com_pos_w[0] - origin),
            "root_position": _serializable_vec(orange.data.root_pos_w[0] - origin),
            "quat_wxyz": _serializable_vec(orange.data.root_quat_w[0]),
        }

    for name in camera_names:
        camera = env.scene[name]
        pos, quat = camera._view.get_world_poses()
        scene["cameras"][name] = {
            "view_position_world": _serializable_vec(pos[0]),
            "view_quat_opengl": _serializable_vec(quat[0]),
        }

    return scene


# Debug aid: when True, perturb_plate_debug() oscillates and tilts the plate each
# step so the plate-oriented in-plate check box (DEBUG_DRAW_PLATE_BOUNDS) can be
# verified to track the plate. This intentionally disrupts the manipulation task —
# for visual debugging of the bounds only.
DEBUG_PERTURB_PLATE = False   # DEBUG: plate wobbles/tilts to verify the check box — turn OFF for real eval

_perturb_anchor: dict = {}


def perturb_plate_debug(env, step_count, plate_name="Plate",
                        xy_amp=0.06, tilt_amp_deg=25.0, period=160):
    """Kinematically wobble and tilt the plate for debugging the oriented check box.

    No-op unless DEBUG_PERTURB_PLATE is True. Re-anchors to the plate's resting
    pose at the start of each episode (step_count <= 0) and writes a smoothly
    oscillating pose every step. Call right before env.step(). The plate is driven
    kinematically (pose + zero velocity each step), so the task is disrupted — this
    is purely to confirm the check box follows the plate's position and tilt.
    """
    if not DEBUG_PERTURB_PLATE:
        return
    import math
    plate = env.scene[plate_name]
    if step_count <= 0 or "pos" not in _perturb_anchor:
        _perturb_anchor["pos"] = plate.data.root_pos_w[0].clone()
    anchor = _perturb_anchor["pos"]
    device = anchor.device
    phase = 2 * math.pi * step_count / period
    new_pos = anchor.clone()
    new_pos[0] += xy_amp * math.sin(phase)
    new_pos[1] += xy_amp * math.cos(0.7 * phase)
    roll  = math.radians(tilt_amp_deg) * math.sin(phase)
    pitch = math.radians(tilt_amp_deg) * math.cos(0.9 * phase)
    quat = _math_utils().quat_from_euler_xyz(
        torch.tensor([roll],  device=device),
        torch.tensor([pitch], device=device),
        torch.tensor([0.0],   device=device),
    )[0]
    pose = torch.cat([new_pos, quat]).unsqueeze(0)
    plate.write_root_pose_to_sim(pose)
    plate.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))


def count_oranges_in_plate(positions,
                           orange_names=("Orange001", "Orange002", "Orange003")):
    """Count orange COMs inside the shared cylindrical plate volume.

    Performs a pure geometric check with no stability requirement — intended as
    a final-state snapshot, not a real-time event detector. If the plate is
    upside down, the scene is a failure and the count is forced to 0.

    Args:
        positions:    dict returned by save_positions(); orange values are COM positions.
        orange_names: scene entity names of the oranges.

    Returns:
        int — number of oranges currently inside the plate bounds.
    """
    if is_plate_upside_down(positions):
        return 0

    count = 0
    for name in orange_names:
        if name in positions and is_orange_position_in_plate(
                positions["plate"], positions[name],
                plate_quat=positions.get("plate_quat")):
            count += 1

    return count


# ============================================================
# Evaluation Tracker
# ============================================================

class EpisodeStory:
    """Compact structured trace of one orchestrated evaluation episode."""

    SCHEMA_VERSION = 1

    def __init__(self, episode, model_id):
        self.episode = int(episode)
        self.model_id = model_id
        self.initial_scene = None
        self.final_scene = None
        self.timeline = []
        self.subtask_attempts = []
        self._active_attempt = None
        self._attempt_id = 0

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if isinstance(value, np.generic):
            return float(value)
        return float(value)

    @classmethod
    def _vec(cls, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        return [round(cls._scalar(v), 5) for v in value]

    @classmethod
    def _scene(cls, plate_pos, orange_positions, labels=None,
               placed_oranges=None, abandoned_oranges=None):
        labels = labels or {}
        placed_oranges = set(placed_oranges or ())
        abandoned_oranges = set(abandoned_oranges or ())
        oranges = {}
        for name, pos in sorted(orange_positions.items()):
            if name in placed_oranges:
                status = "placed"
            elif name in abandoned_oranges:
                status = "abandoned"
            elif placed_oranges or abandoned_oranges:
                status = "unplaced"
            else:
                status = "unknown"
            oranges[name] = {
                "label": labels.get(name),
                "position": cls._vec(pos),
                "status": status,
            }
        return {
            "plate_position": cls._vec(plate_pos) if plate_pos is not None else None,
            "oranges": oranges,
        }

    def record_initial_scene(self, step, plate_pos, orange_positions):
        labels = classify_orange_positions(orange_positions)
        self.initial_scene = self._scene(plate_pos, orange_positions, labels)
        self.add_event(step, "episode_started", phase="SELECT_TARGET")

    def add_event(self, step, event_type, phase=None, requested_orange=None,
                  requested_label=None, actual_orange=None, outcome=None,
                  reason=None, **details):
        event = {
            "step": int(step),
            "event_type": event_type,
            "phase": phase,
            "requested_orange": requested_orange,
            "requested_label": requested_label,
            "actual_orange": actual_orange,
            "outcome": outcome,
            "reason": reason,
        }
        if details:
            event["details"] = details
        self.timeline.append(event)

    def start_attempt(self, step, subtask, prompt, n_placed,
                      requested_orange, requested_label):
        if self._active_attempt is not None:
            self.finish_attempt(
                step,
                result="failure",
                failure_reason="interrupted_by_new_attempt",
            )
        self._attempt_id += 1
        self._active_attempt = {
            "attempt_id": self._attempt_id,
            "subtask": subtask,
            "start_step": int(step),
            "end_step": None,
            "duration_steps": None,
            "prompt": prompt,
            "n_placed_start": int(n_placed),
            "requested_orange": requested_orange,
            "requested_label": requested_label,
            "actual_orange": None,
            "target_match": None,
            "result": None,
            "failure_reason": None,
        }
        self.add_event(
            step,
            f"{subtask.lower()}_started",
            phase=subtask,
            requested_orange=requested_orange,
            requested_label=requested_label,
            outcome="started",
        )

    def finish_attempt(self, step, result, actual_orange=None, failure_reason=None):
        if self._active_attempt is None:
            return
        attempt = self._active_attempt
        requested = attempt["requested_orange"]
        if actual_orange is not None:
            attempt["actual_orange"] = actual_orange
        elif result == "success" and attempt["actual_orange"] is None:
            attempt["actual_orange"] = requested
        if requested is None or attempt["actual_orange"] is None:
            attempt["target_match"] = None
        else:
            attempt["target_match"] = requested == attempt["actual_orange"]
        attempt["end_step"] = int(step)
        attempt["duration_steps"] = int(step) - attempt["start_step"]
        attempt["result"] = result
        attempt["failure_reason"] = failure_reason
        self.subtask_attempts.append(attempt)
        self._active_attempt = None

    def finish_active_as_episode_ended(self, step, reason):
        if self._active_attempt is not None:
            self.finish_attempt(step, result="failure", failure_reason=reason)

    def build_record(self, step_count, oranges_in_plate, end_reason, is_success,
                     plate_pos, orange_positions, placed_oranges,
                     abandoned_oranges):
        self.finish_active_as_episode_ended(step_count, end_reason)
        initial_labels = {}
        if self.initial_scene:
            initial_labels = {
                name: data.get("label")
                for name, data in self.initial_scene.get("oranges", {}).items()
            }
        self.final_scene = self._scene(
            plate_pos,
            orange_positions,
            labels=initial_labels,
            placed_oranges=placed_oranges,
            abandoned_oranges=abandoned_oranges,
        )
        self.add_event(
            step_count,
            "episode_finished",
            outcome="success" if is_success else "failure",
            reason=end_reason,
            oranges_in_plate=int(oranges_in_plate),
        )
        return {
            "episode_summary": {
                "episode": self.episode,
                "model_id": self.model_id,
                "total_steps": int(step_count),
                "final_oranges_in_plate": int(oranges_in_plate),
                "end_reason": end_reason,
                "success": bool(is_success),
            },
            "initial_scene": self.initial_scene,
            "final_scene": self.final_scene,
            "timeline": self.timeline,
            "subtask_attempts": self.subtask_attempts,
        }


class EvaluationTracker:
    """Tracks per-episode outcomes, timing, and oranges placed across a full evaluation run.

    Usage:
        tracker = EvaluationTracker(n_episodes)

        for episode in range(n_episodes):
            tracker.start_episode(episode)
            ...
            tracker.record_timing(infer_ms, step_ms)   # every step
            ...
            tracker.end_episode(episode, steps, terminated, oranges)

        tracker.print_final_summary(model_id)
    """

    # select_action calls below this threshold are queue-replay pops, not real model inference.
    INFER_THRESHOLD_MS = 5.0

    def __init__(
        self,
        n_episodes,
        model_id=None,
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
        self.summary_path = (
            Path(summary_path)
            if summary_path
            else self._default_summary_path(model_id, result_name) if self.checkpoint_path else None
        )
        self.episode_records = []
        self.successes = 0
        self.total_oranges_placed = []
        self.successful_episode_steps = []

        # Per-episode timing buffers, reset at the start of each episode
        self._infer_times = []   # only real inference calls (above INFER_THRESHOLD_MS)
        self._step_times = []

        # Per-subtask attempt/success counts, keyed by "<SUBTASK>_<n_placed>"
        # e.g. "GRASP_0", "LIFT_1", "PLACE_2"
        self._subtask_stats: dict[str, dict[str, int]] = {}

        if resume and self.checkpoint_path:
            self._load_checkpoint()

        # Progress bar advances once per completed episode.
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
        if not cls._short_model_name(model_id):
            return None
        return evaluation_checkpoint_path(model_id, result_name=result_name)

    @classmethod
    def _default_summary_path(cls, model_id, result_name=None):
        if not cls._short_model_name(model_id):
            return None
        return evaluation_summary_path(model_id, result_name=result_name)

    @property
    def next_episode_index(self):
        """Return the next unfinished episode index for resume-aware loops."""
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
            tqdm.write(f"  ⚠ Could not read evaluation checkpoint {self.checkpoint_path}: {exc}")
            return

        checkpoint_model = checkpoint.get("model_id")
        if checkpoint_model and self.model_id and checkpoint_model != self.model_id:
            tqdm.write(
                f"  ⚠ Ignoring evaluation checkpoint for {checkpoint_model}; "
                f"current model is {self.model_id}"
            )
            return

        records = checkpoint.get("episodes", [])
        self.episode_records = sorted(
            (record for record in records if isinstance(record.get("episode"), int)),
            key=lambda record: record["episode"],
        )
        self._recompute_from_records()
        self._subtask_stats = checkpoint.get("subtask_stats", {})
        if self.episode_records:
            tqdm.write(
                f"  ▶ Resuming evaluation metrics: {len(self.episode_records)} "
                f"completed run(s) loaded from {self.checkpoint_path}"
            )

    def _recompute_from_records(self):
        self.total_oranges_placed = [record["oranges_in_plate"] for record in self.episode_records]
        self.successes = sum(1 for record in self.episode_records if record["oranges_in_plate"] == 3)
        self.successful_episode_steps = [
            record["step_count"] for record in self.episode_records if record["oranges_in_plate"] == 3
        ]

    def _summary_text(self, model_id):
        n_eval       = len(self.total_oranges_placed)
        pct          = lambda n: (n / n_eval * 100) if n_eval else 0
        count_3      = self.total_oranges_placed.count(3)
        count_2      = self.total_oranges_placed.count(2)
        count_1      = self.total_oranges_placed.count(1)
        count_0      = self.total_oranges_placed.count(0)
        mean_steps   = (sum(self.successful_episode_steps) / len(self.successful_episode_steps)) if self.successful_episode_steps else float("nan")
        success_rate = (self.successes / n_eval * 100) if n_eval else 0
        avg_oranges  = sum(self.total_oranges_placed) / n_eval if n_eval else 0
        header       = "EVALUATION COMPLETE" if n_eval == self.n_episodes else f"EVALUATION SUMMARY (stopped after {n_eval}/{self.n_episodes} runs)"
        seed_set = seed_set_checkpoint_metadata(self.seed_set)
        seed_line = f"Seed set:             {seed_set['name']} ({seed_set['sha256']})\n" if seed_set else ""

        def fmt(subtask, n):
            s   = self._subtask_stats.get(f"{subtask}_{n}", {})
            att = s.get("attempts", 0)
            suc = s.get("successes", 0)
            p   = suc / att * 100 if att else float("nan")
            return f"{n} placed: {suc:>3}/{att:<3} ({p:5.1f}%)"

        subtask_lines = (
            f"Per-subtask success rates (by oranges in plate at subtask start):\n"
            f"  GRASP:  {fmt('GRASP',0)}   {fmt('GRASP',1)}   {fmt('GRASP',2)}\n"
            f"  LIFT:   {fmt('LIFT', 0)}   {fmt('LIFT', 1)}   {fmt('LIFT', 2)}\n"
            f"  PLACE:  {fmt('PLACE',0)}   {fmt('PLACE',1)}   {fmt('PLACE',2)}\n"
        )

        total_retries     = sum(r.get("n_local_retries",    0) for r in self.episode_records)
        total_redirections= sum(r.get("n_redirections",     0) for r in self.episode_records)
        total_abandoned   = sum(r.get("n_oranges_abandoned",0) for r in self.episode_records)
        eps_with_retry    = sum(1 for r in self.episode_records if r.get("n_local_retries",    0) > 0)
        eps_with_redirect = sum(1 for r in self.episode_records if r.get("n_redirections",     0) > 0)
        mechanism_lines = (
            f"Mechanism usage (local retry = same orange after slip; redirection = new orange after timeout):\n"
            f"  Local retries:      {total_retries} total, fired in {eps_with_retry}/{n_eval} episodes\n"
            f"  Target redirections:{total_redirections} total, fired in {eps_with_redirect}/{n_eval} episodes\n"
            f"  Oranges abandoned:  {total_abandoned} total ({total_abandoned / n_eval:.2f} avg/episode)\n"
        ) if n_eval else ""

        return (
            f"\n========================================\n"
            f"{header}\n"
            f"Model ID:             {model_id}\n"
            f"Result name:          {self.result_name or short_model_name(model_id)}\n"
            f"{seed_line}"
            f"Success Rate:         {self.successes}/{n_eval} ({success_rate:.2f}%)\n"
            f"Avg oranges in plate: {avg_oranges:.2f}/3\n"
            f"Mean steps (success): {mean_steps:.1f}\n"
            f"3/3 oranges:          {count_3}/{n_eval} ({pct(count_3):.1f}%)\n"
            f"2/3 oranges:          {count_2}/{n_eval} ({pct(count_2):.1f}%)\n"
            f"1/3 oranges:          {count_1}/{n_eval} ({pct(count_1):.1f}%)\n"
            f"0/3 oranges:          {count_0}/{n_eval} ({pct(count_0):.1f}%)\n"
            f"Per-episode oranges:  {self.total_oranges_placed}\n"
            f"{subtask_lines}"
            f"{mechanism_lines}"
            f"========================================\n"
        )

    def save_checkpoint(self):
        """Atomically save completed episode metrics so crashes only lose the active run."""
        if not self.checkpoint_path:
            return

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_id": self.model_id,
            "result_name": self.result_name,
            "seed_set": seed_set_checkpoint_metadata(self.seed_set),
            "trace_schema_version": EpisodeStory.SCHEMA_VERSION,
            "target_n_episodes": self.n_episodes,
            "completed_episodes": len(self.episode_records),
            "last_update": datetime.datetime.now().isoformat(timespec="seconds"),
            "episodes": self.episode_records,
            "subtask_stats": self._subtask_stats,
        }
        tmp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(checkpoint, f, indent=2)
        tmp.replace(self.checkpoint_path)

    def write_partial_summary(self, model_id=None):
        if not self.summary_path:
            return
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(self._summary_text(model_id or self.model_id))
        tmp.replace(self.summary_path)

    def record_subtask_result(self, subtask: str, n_placed: int, success: bool):
        """Record one subtask attempt and whether it succeeded."""
        key = f"{subtask}_{n_placed}"
        if key not in self._subtask_stats:
            self._subtask_stats[key] = {"attempts": 0, "successes": 0}
        self._subtask_stats[key]["attempts"] += 1
        if success:
            self._subtask_stats[key]["successes"] += 1

    def start_episode(self, episode):
        """Reset per-episode timing buffers."""
        self._infer_times = []
        self._step_times = []

    def update_step(self, step_count):
        """No-op hook — kept for API compatibility with external inference scripts."""
        pass

    def record_timing(self, infer_time_ms, step_time_ms):
        """Append timing samples.

        infer_time_ms values below INFER_THRESHOLD_MS are queue-replay pops and are
        discarded so the average only reflects actual model forward passes.
        None values are ignored.
        """
        if infer_time_ms is not None and infer_time_ms > self.INFER_THRESHOLD_MS:
            self._infer_times.append(infer_time_ms)
        if step_time_ms is not None:
            self._step_times.append(step_time_ms)

    def end_episode(self, episode, step_count, is_terminated, oranges_in_plate,
                    n_local_retries=0, n_redirections=0, n_oranges_abandoned=0,
                    camera_images=None, episode_story=None,
                    seed_metadata=None, initial_scene_audit=None,
                    start_camera_images=None):
        """Record episode result, print a summary line, and advance the progress bar."""
        n_infer_calls  = len(self._infer_times)
        last_infer     = self._infer_times[-1] if self._infer_times else float("nan")
        avg_step       = (sum(self._step_times) / len(self._step_times)) if self._step_times else float("nan")
        outcome        = "TERMINATED" if is_terminated else "TRUNCATED"
        was_new_episode = all(record["episode"] != episode for record in self.episode_records)

        record = {
            "episode": episode,
            "step_count": step_count,
            "is_terminated": bool(is_terminated),
            "oranges_in_plate": int(oranges_in_plate),
            "n_infer_calls": n_infer_calls,
            "last_infer_ms": last_infer,
            "avg_step_ms": avg_step,
            "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_local_retries": int(n_local_retries),
            "n_redirections": int(n_redirections),
            "n_oranges_abandoned": int(n_oranges_abandoned),
        }
        if seed_metadata:
            record.update(seed_metadata)
        if initial_scene_audit:
            record["initial_scene_audit"] = initial_scene_audit
        if episode_story:
            record.update(episode_story)
        self.episode_records = [r for r in self.episode_records if r["episode"] != episode]
        self.episode_records.append(record)
        self.episode_records.sort(key=lambda r: r["episode"])
        self._recompute_from_records()

        mech = ""
        if n_local_retries or n_redirections:
            mech = f" | Retries: {n_local_retries}  Redirections: {n_redirections}"
        tqdm.write(
            f"  Episode {episode:>3d} | {outcome:<10s} | "
            f"Oranges: {oranges_in_plate}/3 | "
            f"Steps: {step_count:>4d} | "
            f"Infer: {n_infer_calls} real calls, last {last_infer:.0f} ms | "
            f"Avg step: {avg_step:>6.1f} ms{mech}"
        )
        if was_new_episode:
            self._pbar.update(1)
        if start_camera_images:
            try:
                save_episode_camera_snapshots(
                    self.model_id,
                    "autonomous",
                    episode,
                    start_camera_images.get("front"),
                    start_camera_images.get("wrist"),
                    result_name=self.result_name,
                    stage="start",
                )
            except Exception as exc:
                tqdm.write(f"  ⚠ Could not save episode start camera snapshots: {exc}")
        if camera_images:
            try:
                save_episode_camera_snapshots(
                    self.model_id,
                    "autonomous",
                    episode,
                    camera_images.get("front"),
                    camera_images.get("wrist"),
                    result_name=self.result_name,
                )
            except Exception as exc:
                tqdm.write(f"  ⚠ Could not save episode camera snapshots: {exc}")
        self.save_checkpoint()
        self.write_partial_summary()

    def print_final_summary(self, model_id):
        """Print the evaluation summary and update results/_latest.txt."""
        self._pbar.close()
        print(self._summary_text(model_id))
        self.write_partial_summary(model_id)
        if self.summary_path:
            print(f"Summary saved to: {self.summary_path}\n")


# ============================================================
# Subtask Tracker
# ============================================================

class SubtaskTracker:
    """Detects three robot subtask events each step and live-displays their conditions.

    Events detected:
      1. Grasp — Gripper is closed and the EE is well-centred over an unplaced orange
                 for N consecutive frames.
      2. Lift  — Gripper is closed and the active orange has been raised above its
                 initial height for N consecutive frames.
      3. Place — The active orange has been stably inside the plate for N consecutive
                 frames while the gripper is open (i.e. it has been released).

    Grasp sets the active orange; Lift and Place operate only on that orange.
    Each confirmation fires a single persistent print and then goes silent.
    """

    # Plate bounds relative to plate centre (m): truncated cone then cylinder.
    PLATE_RADIUS       = PLATE_RADIUS        # XY radius of the cylindrical upper section
    PLATE_INNER_RADIUS = PLATE_INNER_RADIUS  # XY radius at the bottom of the interior
    PLATE_CONE_HEIGHT  = PLATE_CONE_HEIGHT   # height of the cone section above PLATE_Z_MIN
    PLATE_Z_MIN  = PLATE_Z_MIN   # bottom of the volume (relative to plate centre Z)
    PLATE_Z_MAX  = PLATE_Z_MAX   # top of the volume

    DEBUG_DRAW            = False  # set to True to visualize the grip axis in the Isaac Sim viewport
    DEBUG_DRAW_PLATE_BOUNDS = True   # DEBUG: draw the plate occupancy cylinder — turn OFF for real eval
    ORANGE_HELD_MAX_DIST  = 0.06   # max tip-to-orange distance (m) to consider orange still held
    PLACE_GRIPPER_Z_MIN   = 0.04   # gripper tip must be at or above this env-relative Z to confirm place
    GRASP_APPROACH_DIST   = 0.10   # tip-to-orange distance (m) beyond which only V_approach contributes

    def __init__(
        self,
        block = False,              # if True, pause after each confirmed event
        patience_frames=10,         # consecutive frames required to confirm grasp or lift
        centering_threshold=0.015,   # info only — not used as confirmation condition
        closure_threshold=0.065,    # info only — not used as confirmation condition
        grip_t_min=0.4,             # info only — not used as confirmation condition
        grip_t_max=0.67,            # info only — not used as confirmation condition
        contact_force_min=10.0,     # min projected contact force (N) on each tip to confirm grasp
        grasp_threshold=0.60,       # gripper joint value: above = open, below = closed (used by lift/place)
        lift_height_threshold=0.06,  # min height gain from initial Z to confirm lift (m)
        orange_names=("Orange001", "Orange002", "Orange003"),
        stability_frames=10,        # frames orange must be stationary inside plate to confirm place
        stability_tolerance=0.001,  # max orange movement per frame to count as stationary (m)
    ):
        self.patience_frames       = patience_frames
        self.centering_threshold   = centering_threshold
        self.closure_threshold     = closure_threshold
        self.grip_t_min            = grip_t_min
        self.grip_t_max            = grip_t_max
        self.contact_force_min     = contact_force_min
        self.grasp_threshold       = grasp_threshold
        self.lift_height_threshold = lift_height_threshold
        self.orange_names          = orange_names
        self.total_oranges         = len(orange_names)
        self.stability_frames      = stability_frames
        self.stability_tolerance   = stability_tolerance
        self.block = block
        self.reset()

    def reset(self):
        """Reset all state for a new episode."""
        self.placed_oranges    = set()
        self.initial_orange_z  = {}
        self.active_orange     = None    # orange currently being worked on
        self.grasp_counter     = 0
        self.lift_counter      = 0
        self._grasp_confirmed  = False
        self._lift_confirmed   = False
        self._place_confirmed  = False
        self._status_lines     = 0       # lines currently held by the live display block
        self._origin           = None    # cached env origin for debug drawing
        self._plate_pos        = None    # cached plate position for debug drawing
        self._plate_quat       = None    # cached plate orientation (w,x,y,z) for oriented check
        self._gripper_tip      = None    # cached tip positions for held-check in lift/place
        self._jaw_tip          = None
        self._stability: dict[str, tuple[int, torch.Tensor | None]] = {}

    def reset_grasp_state(self):
        """Reset active orange and all subtask flags without clearing placed_oranges or initial heights.
        Call each time the user re-issues a 'grasp' prompt for a fresh orange selection.
        """
        self.active_orange    = None
        self.grasp_counter    = 0
        self.lift_counter     = 0
        self._grasp_confirmed = False
        self._lift_confirmed  = False
        self._place_confirmed = False
        self._status_lines    = 0

    def reset_display(self):
        """Reset cursor tracking so the next live update starts fresh below the current cursor."""
        self._status_lines = 0

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _get_env_data(self, env):
        """Extract all needed scene quantities, normalised to env origin."""
        self._origin = env.scene.env_origins[0]   # stored for debug drawing
        ee_frame    = env.scene["ee_frame"]
        frame_names = ee_frame.data.target_frame_names
        frames      = ee_frame.data.target_pos_w[0]
        gripper_tip       = frames[frame_names.index("gripper_tip")] - self._origin
        jaw_tip           = frames[frame_names.index("jaw_tip")]     - self._origin
        self._gripper_tip = gripper_tip
        self._jaw_tip     = jaw_tip
        gripper_pos = env.scene["robot"].data.joint_pos[0, -1].item()
        plate_pos         = env.scene["Plate"].data.root_pos_w[0] - self._origin
        self._plate_pos   = plate_pos
        self._plate_quat  = env.scene["Plate"].data.root_quat_w[0]
        orange_positions = {
            name: env.scene[name].data.root_com_pos_w[0] - self._origin
            for name in self.orange_names
        }
        gripper_forces = env.scene["gripper_contact"].data.net_forces_w[0]  # (num_bodies, 3)
        jaw_forces     = env.scene["jaw_contact"].data.net_forces_w[0]
        gripper_force_vec = gripper_forces[gripper_forces.norm(dim=-1).argmax()].cpu()
        jaw_force_vec     = jaw_forces[jaw_forces.norm(dim=-1).argmax()].cpu()
        return gripper_tip, jaw_tip, gripper_pos, gripper_force_vec, jaw_force_vec, plate_pos, orange_positions

    def _pause(self):
        if(self.block):
            input("  ⏸  Press Enter to continue...")

    def _live_update(self, lines):
        """Overwrite the previous status block in-place using ANSI escape codes."""
        if self._status_lines > 0:
            sys.stdout.write(f"\033[{self._status_lines}A")
        for line in lines:
            sys.stdout.write(f"\r\033[2K{line}\n")
        sys.stdout.flush()
        self._status_lines = len(lines)

    def _is_orange_in_plate(self, orange_pos) -> bool:
        """Return True if orange_pos is inside the cylindrical plate bounds."""
        if self._plate_pos is None:
            return False
        return is_orange_position_in_plate(
            self._plate_pos,
            orange_pos,
            plate_radius=self.PLATE_RADIUS,
            plate_z_min=self.PLATE_Z_MIN,
            plate_z_max=self.PLATE_Z_MAX,
            plate_quat=self._plate_quat,
            inner_radius=self.PLATE_INNER_RADIUS,
            cone_height=self.PLATE_CONE_HEIGHT,
        )

    def _is_orange_held(self, orange_pos) -> tuple[bool, float]:
        """Return (held, min_dist) — True if the closest tip is within ORANGE_HELD_MAX_DIST."""
        if self._gripper_tip is None or self._jaw_tip is None:
            return True, 0.0
        d_gripper = (self._gripper_tip - orange_pos).norm().item()
        d_jaw     = (self._jaw_tip     - orange_pos).norm().item()
        min_dist  = min(d_gripper, d_jaw)
        return min_dist < self.ORANGE_HELD_MAX_DIST, min_dist

    def _draw_grip_axis(self, gripper_tip, jaw_tip, best_orange_pos, best_proj, meets):
        """Draw the grip axis and centering debug geometry in the Isaac Sim viewport.

        Draws (in world space, cleared each call):
          - Grip axis segment gripper_tip → jaw_tip  (green = all conditions met, red otherwise)
          - Line from orange centre to its projection on the axis  (yellow)
          - Dot at the projection point  (yellow)
        """
        if not self.DEBUG_DRAW or self._origin is None:
            return
        try:
            import omni.debugdraw
            import carb
            draw = omni.debugdraw.get_debug_draw_interface()
        except Exception:
            return

        def w(pos):
            p = pos + self._origin
            return carb.Float3(p[0].item(), p[1].item(), p[2].item())

        axis_color = 0xFF00FF00 if meets else 0xFF0000FF   # green / blue  (ARGB)
        draw.draw_line(w(gripper_tip), axis_color, 4.0, w(jaw_tip), axis_color, 4.0)

        if best_orange_pos is not None and best_proj is not None:
            draw.draw_line(w(best_orange_pos), 0xFF00FFFF, 2.0, w(best_proj), 0xFF00FFFF, 2.0)
            draw.draw_point(w(best_proj), 0xFF00FFFF, 10.0)


    def _draw_plate_cylinder(self):
        """Draw the plate check volume (truncated cone + cylinder) in the viewport."""
        if not (self.DEBUG_DRAW or self.DEBUG_DRAW_PLATE_BOUNDS) or self._origin is None or self._plate_pos is None:
            return
        try:
            import omni.debugdraw
            import carb
            import math
            draw = omni.debugdraw.get_debug_draw_interface()
        except Exception:
            return

        base = self._plate_pos + self._origin
        q    = self._plate_quat
        z0 = self.PLATE_Z_MIN
        z1 = self.PLATE_Z_MAX
        z_cone = z0 + self.PLATE_CONE_HEIGHT
        color, center_color, w, N = 0xFFFF8800, 0xFFFFFF00, 2.5, 48  # orange/yellow, line width, segments

        # Ring levels as (z, radius): bottom of cone, cone top, cylinder top.
        # The cone-top ring is only added when the cone fits below PLATE_Z_MAX.
        levels = [(z0, self.PLATE_INNER_RADIUS)]
        if z0 < z_cone < z1:
            levels.append((z_cone, self.PLATE_RADIUS))
        levels.append((z1, self.PLATE_RADIUS))

        def to_world(lx, ly, lz):
            """Map a plate-local offset to a world-space carb.Float3, applying plate tilt."""
            v = torch.tensor([lx, ly, lz], dtype=base.dtype, device=base.device)
            if q is not None:
                v = _math_utils().quat_apply(q, v)
            v = v + base
            return carb.Float3(v[0].item(), v[1].item(), v[2].item())

        rings = []
        for z, r in levels:
            rings.append([
                to_world(r * math.cos(2 * math.pi * i / N),
                         r * math.sin(2 * math.pi * i / N), z)
                for i in range(N)
            ])

        center_bottom = to_world(0.0, 0.0, z0)
        center_top    = to_world(0.0, 0.0, z1)
        draw.draw_line(center_bottom, center_color, w, center_top, center_color, w)

        for ring in rings:
            for i in range(N):
                draw.draw_line(ring[i], color, w, ring[(i + 1) % N], color, w)

        # Vertical/sloped wall connectors traced through every ring level.
        for i in range(0, N, 4):
            for k in range(len(rings) - 1):
                draw.draw_line(rings[k][i], color, w, rings[k + 1][i], color, w)
            draw.draw_line(center_bottom, color, 1.5, rings[0][i], color, 1.5)
            draw.draw_line(center_top, color, 1.5, rings[-1][i], color, 1.5)

    def draw_debug(self, gripper_tip, jaw_tip, orange_positions):
        """Draw grip axis debug geometry every step regardless of active prompt."""
        axis    = jaw_tip - gripper_tip
        axis_sq = torch.dot(axis, axis).item()

        target = self.active_orange
        candidates = {target: orange_positions[target]} if target and target in orange_positions else orange_positions

        best_name, best_dist, best_t = None, float("inf"), 0.0
        for name, pos in candidates.items():
            if name in self.placed_oranges or name not in self.initial_orange_z:
                continue
            t_raw     = torch.dot(pos - gripper_tip, axis).item() / (axis_sq + 1e-8)
            t_clamped = max(0.0, min(1.0, t_raw))
            proj      = gripper_tip + t_clamped * axis
            dist      = (pos - proj).norm().item()
            if dist < best_dist:
                best_dist, best_name, best_t = dist, name, t_raw

        best_proj = None
        if best_name is not None:
            best_proj = gripper_tip + max(0.0, min(1.0, best_t)) * axis

        gap   = axis_sq ** 0.5
        t_ok  = best_name is not None and self.grip_t_min <= best_t <= self.grip_t_max
        meets = (best_name is not None
                 and best_dist < self.centering_threshold
                 and gap < self.closure_threshold
                 and t_ok)

        self._draw_grip_axis(gripper_tip, jaw_tip, orange_positions.get(best_name), best_proj, meets)
        self._draw_plate_cylinder()

    # ----------------------------------------------------------
    # Main entry point — call every step
    # ----------------------------------------------------------

    def check_status(self, env, step_count):
        """Run all subtask checks for the current step."""
        gripper_tip, jaw_tip, gripper_pos, gripper_force_vec, jaw_force_vec, plate_pos, orange_positions = self._get_env_data(env)

        if step_count == 0:
            for name, pos in orange_positions.items():
                self.initial_orange_z[name] = pos[2].item()

        self._check_grasp(gripper_tip, jaw_tip, orange_positions, step_count, gripper_force_vec, jaw_force_vec)
        self._check_lift(gripper_pos, orange_positions, step_count)
        self._check_place(plate_pos, orange_positions, gripper_pos, step_count)

    # ----------------------------------------------------------
    # 1. Grasp check
    # ----------------------------------------------------------

    def _check_grasp(self, gripper_tip, jaw_tip, orange_positions, step_count, gripper_force_vec=None, jaw_force_vec=None):
        """Live-display grasp conditions each step; confirm once patience is reached.

        Centering: distance from orange centre to the gripper_tip→jaw_tip segment.
        Closure:   distance between the two fingertips (finger gap).
        Both must be under threshold for patience_frames consecutive steps.
        Always tracks the closest unplaced orange to show live feedback before active_orange is set.
        """
        if self._grasp_confirmed:
            return

        axis      = jaw_tip - gripper_tip
        axis_sq   = torch.dot(axis, axis).item()
        gap       = axis_sq ** 0.5
        axis_unit = axis / (gap + 1e-8)
        gripper_grasp_N = abs(torch.dot(gripper_force_vec.to(axis_unit.device), axis_unit).item()) if gripper_force_vec is not None else 0.0
        jaw_grasp_N     = abs(torch.dot(jaw_force_vec.to(axis_unit.device),     axis_unit).item()) if jaw_force_vec     is not None else 0.0

        # Find the closest unplaced orange to the grip axis
        best_name = None
        best_dist = float("inf")
        best_t    = 0.0
        for name, orange_pos in orange_positions.items():
            if name in self.placed_oranges or name not in self.initial_orange_z:
                continue
            t_raw = torch.dot(orange_pos - gripper_tip, axis).item() / (axis_sq + 1e-8)
            t_clamped = max(0.0, min(1.0, t_raw))
            proj = gripper_tip + t_clamped * axis
            dist = (orange_pos - proj).norm().item()
            if dist < best_dist:
                best_dist, best_name, best_t = dist, name, t_raw

        t_ok  = best_name is not None and self.grip_t_min <= best_t <= self.grip_t_max
        meets = (best_name is not None
                 and best_dist < self.centering_threshold
                 and gap < self.closure_threshold
                 and t_ok
                 and gripper_grasp_N >= self.contact_force_min
                 and jaw_grasp_N     >= self.contact_force_min)

        if meets:
            self.grasp_counter += 1
        else:
            self.grasp_counter = 0

        # Orange position label
        pos_labels   = classify_orange_positions(orange_positions)
        orange_label = f"{best_name} / {pos_labels.get(best_name, '?')}" if best_name else "?"

        # Live display
        c_sym  = "✓" if best_dist < self.centering_threshold else "✗"
        g_sym  = "✓" if gap < self.closure_threshold else "✗"
        t_sym  = "✓" if t_ok else "✗"
        gf_sym = "✓" if gripper_grasp_N >= self.contact_force_min else "✗"
        jf_sym = "✓" if jaw_grasp_N     >= self.contact_force_min else "✗"
        lines = [
            f"  ✊ GRASP [{orange_label}]  step {step_count}",
            f"     Centering: {best_dist:.4f} < {self.centering_threshold}  {c_sym}",
            f"     Closure:   {gap:.4f} < {self.closure_threshold}  {g_sym}",
            f"     Grip pos:  t={best_t:.2f}  ∈ [{self.grip_t_min}, {self.grip_t_max}]  {t_sym}",
            f"     Gripper F: {gripper_grasp_N:.2f} N >= {self.contact_force_min}  {gf_sym}",
            f"     Jaw F:     {jaw_grasp_N:.2f} N >= {self.contact_force_min}  {jf_sym}",
            f"     Patience:  {self.grasp_counter}/{self.patience_frames}",
        ]
        self._live_update(lines)

        if self.grasp_counter == self.patience_frames:
            self._grasp_confirmed = True
            self._place_confirmed = False
            self.active_orange    = best_name
            self._status_lines    = 0
            print(f"  ✊ GRASP CONFIRMED: {self.active_orange} is now the active orange  (step {step_count})\n")
            self._pause()

    # ----------------------------------------------------------
    # 2. Lift check
    # ----------------------------------------------------------

    def _check_lift(self, gripper_pos, orange_positions, step_count):
        """Live-display lift conditions for the active orange each step."""
        if self._lift_confirmed:
            return

        gripper_closed   = gripper_pos < self.grasp_threshold
        currently_lifted = None

        target = self.active_orange
        candidates = (
            {target: orange_positions[target]}
            if target and target in orange_positions
            else orange_positions
        )

        if gripper_closed:
            for name, pos in candidates.items():
                if name in self.placed_oranges or name not in self.initial_orange_z:
                    continue
                if pos[2].item() > self.initial_orange_z[name] + self.lift_height_threshold:
                    currently_lifted = (name, pos, pos[2].item(), self.initial_orange_z[name])
                    break

        if currently_lifted:
            self.lift_counter += 1
        else:
            self.lift_counter = 0

        display = target if target else "?"
        g_sym   = "✓" if gripper_closed else "✗"
        if target and target in orange_positions:
            pos         = orange_positions[target]
            height_gain = pos[2].item() - self.initial_orange_z.get(target, pos[2].item())
            h_sym = "✓" if height_gain > self.lift_height_threshold else "✗"
            lines = [
                f"  🤏 LIFT [{display}]  step {step_count}",
                f"     Gripper:     {gripper_pos:.4f} < {self.grasp_threshold} (closed)  {g_sym}",
                f"     Height gain: {height_gain:.4f} > {self.lift_height_threshold}  {h_sym}",
                f"     Patience:    {self.lift_counter}/{self.patience_frames}",
            ]
        else:
            lines = [
                f"  🤏 LIFT  step {step_count}  (no active orange — grasp first)",
                f"     Gripper:  {gripper_pos:.4f} < {self.grasp_threshold} (closed)  {g_sym}",
            ]
        self._live_update(lines)

        if self.lift_counter == self.patience_frames:
            name, pos, current_z, initial_z = currently_lifted
            self._lift_confirmed = True
            self._status_lines   = 0
            print(f"  🤏 LIFT CONFIRMED: {name} lifted  (height gain: {current_z - initial_z:.4f} m,  step {step_count})\n")
            self._pause()

    # ----------------------------------------------------------
    # 3. Place check
    # ----------------------------------------------------------

    def _check_place(self, plate_pos, orange_positions, gripper_pos, step_count):
        """Live-display place conditions for the active orange each step."""
        if self._place_confirmed:
            return

        pz = plate_pos[2].item()
        gripper_open = gripper_pos > self.grasp_threshold

        target     = self.active_orange
        candidates = (
            {target: orange_positions[target]}
            if target and target in orange_positions
            else orange_positions
        )

        newly_confirmed = []
        for name, opos in candidates.items():
            if name in self.placed_oranges:
                continue
            _xy_dist, _z_offset, in_plate = plate_position_metrics(
                plate_pos,
                opos,
                plate_radius=self.PLATE_RADIUS,
                plate_z_min=self.PLATE_Z_MIN,
                plate_z_max=self.PLATE_Z_MAX,
                plate_quat=self._plate_quat,
                inner_radius=self.PLATE_INNER_RADIUS,
                cone_height=self.PLATE_CONE_HEIGHT,
            )
            if not in_plate:
                self._stability[name] = (0, None)
                continue
            prev_frames, prev_pos = self._stability.get(name, (0, None))
            moved = (
                (opos - prev_pos).norm().item() > self.stability_tolerance
                if prev_pos is not None else False
            )
            stable_frames = 0 if moved else prev_frames + 1
            self._stability[name] = (stable_frames, opos.clone())
            gripper_z = (self._gripper_tip[2].item() - pz) if self._gripper_tip is not None else float("inf")
            if stable_frames >= self.stability_frames and gripper_open and gripper_z >= self.PLACE_GRIPPER_Z_MIN:
                newly_confirmed.append((name, opos, stable_frames))
                self.placed_oranges.add(name)

        # Live status
        display = target if target else "?"
        g_sym   = "✓" if gripper_open else "✗"
        if target and target in orange_positions:
            opos  = orange_positions[target]
            xy_dist, z_offset, in_plate = plate_position_metrics(
                plate_pos,
                opos,
                plate_radius=self.PLATE_RADIUS,
                plate_z_min=self.PLATE_Z_MIN,
                plate_z_max=self.PLATE_Z_MAX,
                plate_quat=self._plate_quat,
                inner_radius=self.PLATE_INNER_RADIUS,
                cone_height=self.PLATE_CONE_HEIGHT,
            )
            stable_frames, _ = self._stability.get(target, (0, None))
            held, held_dist  = self._is_orange_held(opos)
            gripper_z = (self._gripper_tip[2].item() - pz) if self._gripper_tip is not None else float("nan")
            allowed_r = plate_radius_at_height(
                z_offset, self.PLATE_RADIUS, self.PLATE_INNER_RADIUS,
                self.PLATE_CONE_HEIGHT, self.PLATE_Z_MIN,
            )
            r_sym  = "✓" if xy_dist < allowed_r else "✗"
            z_sym  = "✓" if self.PLATE_Z_MIN < z_offset < self.PLATE_Z_MAX else "✗"
            s_sym  = "✓" if stable_frames >= self.stability_frames else "✗"
            d_sym  = "✓" if held else "✗"
            gz_sym = "✓" if gripper_z >= self.PLACE_GRIPPER_Z_MIN else "✗"
            g_sym  = "✓" if gripper_open else "✗"
            lines = [
                f"  🍊 PLACE [{display}]  step {step_count}",
                f"     Held:       {held_dist:.4f} < {self.ORANGE_HELD_MAX_DIST}  {d_sym}  (info)",
                f"     XY dist:    {xy_dist:.4f} < {allowed_r:.4f}  {r_sym}",
                f"     Orange Z:   {z_offset:.4f}  ∈ [{self.PLATE_Z_MIN}, {self.PLATE_Z_MAX}]  {z_sym}",
                f"     Tip-Plate Z:{gripper_z:.4f} >= {self.PLACE_GRIPPER_Z_MIN}  {gz_sym}",
                f"     Stable:     {stable_frames}/{self.stability_frames}  {s_sym}",
                f"     Open:       {gripper_pos:.4f} > {self.grasp_threshold}  {g_sym}",
            ]
        else:
            lines = [
                f"  🍊 PLACE  step {step_count}  (no active orange — grasp first)",
                f"     Gripper:  {gripper_pos:.4f} > {self.grasp_threshold} (open)  {g_sym}",
            ]
        self._live_update(lines)

        if newly_confirmed:
            confirmed_names = ", ".join(n for n, _, _ in newly_confirmed)
            current_count   = len(self.placed_oranges)
            self.active_orange    = None  # ready for next orange
            self._grasp_confirmed = False
            self._lift_confirmed  = False
            self._place_confirmed = True
            self._status_lines    = 0
            print(f"  🍊 PLACE CONFIRMED: {confirmed_names}  ({current_count}/{self.total_oranges} total,  step {step_count})\n")
            self._pause()


# ============================================================
# Home Position Checker
# ============================================================

class HomeChecker:
    """Detects when the robot has returned to its episode-start joint configuration.

    Fires once per episode when all joints are within `joint_threshold` radians of
    their initial positions for `patience_frames` consecutive frames.
    """

    def __init__(self, patience_frames=5):
        self.patience_frames = patience_frames
        self._counter        = 0
        self._fired          = False
        self._status_lines   = 0

    def reset(self):
        self._counter      = 0
        self._fired        = False
        self._status_lines = 0

    def reset_display(self):
        self._status_lines = 0

    def _live_update(self, lines):
        if self._status_lines > 0:
            sys.stdout.write(f"\033[{self._status_lines}A")
        for line in lines:
            sys.stdout.write(f"\r\033[2K{line}\n")
        sys.stdout.flush()
        self._status_lines = len(lines)

    def check(self, env, step_count: int):
        if self._fired:
            return

        joint_pos   = env.scene["robot"].data.joint_pos[0]
        joint_names = env.scene["robot"].data.joint_names
        joint_deg   = joint_pos.cpu() / torch.pi * 180.0

        at_rest = all(
            lo < joint_deg[joint_names.index(name)].item() < hi
            for name, (lo, hi) in _SO101_REST_POSE_DEG.items()
        )

        if at_rest:
            self._counter += 1
        else:
            self._counter = 0

        lines = [f"  🏠 HOME  step {step_count}"]
        for joint_name, (lo, hi) in _SO101_REST_POSE_DEG.items():
            idx = joint_names.index(joint_name)
            val = joint_deg[idx].item()
            ok  = "✓" if lo < val < hi else "✗"
            lines.append(f"     {joint_name:<14s} {val:+7.1f}°  ∈ [{lo:.0f}, {hi:.0f}]  {ok}")
        lines.append(f"     Patience:      {self._counter}/{self.patience_frames}")
        self._live_update(lines)

        if self._counter == self.patience_frames:
            self._fired        = True
            self._status_lines = 0
            print(f"  🏠 HOME CONFIRMED at step {step_count}\n")
