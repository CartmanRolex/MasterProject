"""
Evaluation utilities for robotic pick-and-place tasks.

Provides:
  - save_positions / count_oranges_in_plate : geometric scene-state helpers
  - save_camera_snapshots                   : debug image dumps
  - EvaluationTracker                       : per-episode logging, tqdm bar, final summary + file export
  - SubtaskTracker                          : fine-grained phase tracker (multi-object pick-and-place)
"""

import os
import sys
import datetime

# Rest-pose range for SO101 (degrees). Mirrors SO101_FOLLOWER_REST_POSE_RANGE in leisaac.
_SO101_REST_POSE_DEG = {
    "shoulder_pan":  (-30.0,   30.0),   # 0°
    "shoulder_lift": (-130.0, -70.0),   # -100°
    "elbow_flex":    (  60.0, 120.0),   # 90°
    "wrist_flex":    (  20.0,  80.0),   # 50°
    "wrist_roll":    ( -30.0,  30.0),   # 0°
    "gripper":       ( -40.0,  20.0),   # -10°
}

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


# ============================================================
# Camera Debugging
# ============================================================

def save_camera_snapshots(raw_front, raw_wrist, episode, step_count,
                          target_step=15, target_episode=0):
    """Save front/wrist camera images at a specific (episode, step) for visual sanity-checking."""
    if episode != target_episode or step_count != target_step:
        return

    for name, img in [("front", raw_front), ("wrist", raw_wrist)]:
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3:  # CHW -> HWC
            img = img.transpose(1, 2, 0)
        Image.fromarray(img).save(f"camera_check_{name}.png")
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

def save_positions(env, plate_name="Plate",
                   orange_names=("Orange001", "Orange002", "Orange003")):
    """Snapshot current world positions of the plate and all oranges, relative to env origin.

    Should be called every step. The last snapshot before done=True reflects the
    true final state before the env auto-resets.

    Returns:
        dict mapping "plate" and each orange name to a cloned position tensor.
    """
    origin = env.scene.env_origins[0]
    positions = {"plate": env.scene[plate_name].data.root_pos_w[0].clone() - origin}
    for name in orange_names:
        positions[name] = env.scene[name].data.root_pos_w[0].clone() - origin
    return positions


def count_oranges_in_plate(positions,
                           orange_names=("Orange001", "Orange002", "Orange003"),
                           x_range=(-0.10, 0.10),
                           y_range=(-0.10, 0.10),
                           height_range=(-0.01, 0.05)):
    """Count how many oranges are currently within the plate bounds.

    Performs a pure geometric check with no stability requirement — intended as
    a final-state snapshot, not a real-time event detector.

    Args:
        positions:    dict returned by save_positions().
        orange_names: scene entity names of the oranges.
        x_range:      acceptable x offset from plate centre (m).
        y_range:      acceptable y offset from plate centre (m).
        height_range: acceptable z offset from plate centre (m).

    Returns:
        int — number of oranges currently inside the plate bounds.
    """
    px, py, pz = (positions["plate"][i] for i in range(3))

    count = 0
    for name in orange_names:
        ox, oy, oz = (positions[name][i] for i in range(3))
        in_plate = (
            (px + x_range[0] < ox < px + x_range[1])
            and (py + y_range[0] < oy < py + y_range[1])
            and (pz + height_range[0] < oz < pz + height_range[1])
        )
        if in_plate:
            count += 1

    return count


# ============================================================
# Evaluation Tracker
# ============================================================

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

    def __init__(self, n_episodes):
        self.n_episodes = n_episodes
        self.successes = 0
        self.total_oranges_placed = []
        self.successful_episode_steps = []

        # Per-episode timing buffers, reset at the start of each episode
        self._infer_times = []
        self._step_times = []

        # Progress bar advances once per episode
        self._pbar = tqdm(total=n_episodes, desc="Episodes", unit="ep")

    def start_episode(self, episode):
        """Reset per-episode timing buffers."""
        self._infer_times = []
        self._step_times = []

    def update_step(self, step_count):
        """No-op hook — kept for API compatibility with external inference scripts."""
        pass

    def record_timing(self, infer_time_ms, step_time_ms):
        """Append timing samples. None values (e.g. ACT queue-replay steps) are ignored."""
        if infer_time_ms is not None:
            self._infer_times.append(infer_time_ms)
        if step_time_ms is not None:
            self._step_times.append(step_time_ms)

    def end_episode(self, episode, step_count, is_terminated, oranges_in_plate):
        """Record episode result, print a summary line, and advance the progress bar."""
        self.total_oranges_placed.append(oranges_in_plate)
        if is_terminated:
            self.successes += 1
            self.successful_episode_steps.append(step_count)

        avg_infer = (sum(self._infer_times) / len(self._infer_times)) if self._infer_times else float("nan")
        avg_step  = (sum(self._step_times)  / len(self._step_times))  if self._step_times  else float("nan")
        outcome   = "TERMINATED" if is_terminated else "TRUNCATED"

        tqdm.write(
            f"  Episode {episode:>3d} | {outcome:<10s} | "
            f"Oranges: {oranges_in_plate}/3 | "
            f"Steps: {step_count:>4d} | "
            f"Avg infer: {avg_infer:>6.1f} ms | "
            f"Avg step: {avg_step:>6.1f} ms"
        )
        self._pbar.update(1)

    def print_final_summary(self, model_id):
        """Print and save the evaluation summary to results/."""
        self._pbar.close()

        n_eval       = len(self.total_oranges_placed)
        pct          = lambda n: (n / n_eval * 100) if n_eval else 0
        count_3      = self.total_oranges_placed.count(3)
        count_2      = self.total_oranges_placed.count(2)
        count_1      = self.total_oranges_placed.count(1)
        count_0      = self.total_oranges_placed.count(0)
        mean_steps   = (sum(self.successful_episode_steps) / len(self.successful_episode_steps)) if self.successful_episode_steps else float("nan")
        success_rate = (self.successes / self.n_episodes * 100) if self.n_episodes else 0
        avg_oranges  = sum(self.total_oranges_placed) / n_eval if n_eval else 0

        summary_text = (
            f"\n========================================\n"
            f"EVALUATION COMPLETE\n"
            f"Model ID:             {model_id}\n"
            f"Success Rate:         {self.successes}/{self.n_episodes} ({success_rate:.2f}%)\n"
            f"Avg oranges in plate: {avg_oranges:.2f}/3\n"
            f"Mean steps (success): {mean_steps:.1f}\n"
            f"3/3 oranges:          {count_3}/{n_eval} ({pct(count_3):.1f}%)\n"
            f"2/3 oranges:          {count_2}/{n_eval} ({pct(count_2):.1f}%)\n"
            f"1/3 oranges:          {count_1}/{n_eval} ({pct(count_1):.1f}%)\n"
            f"0/3 oranges:          {count_0}/{n_eval} ({pct(count_0):.1f}%)\n"
            f"Per-episode oranges:  {self.total_oranges_placed}\n"
            f"========================================\n"
        )
        print(summary_text)

        os.makedirs("results", exist_ok=True)
        timestamp        = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_model_name = model_id.rstrip("/").split("/")[-1]
        filename         = f"results/eval_{short_model_name}_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write(summary_text)
        print(f"Summary saved to: {filename}\n")


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

    # Cylindrical plate bounds relative to plate centre (m).
    PLATE_RADIUS = 0.10    # XY radius of the cylinder
    PLATE_Z_MIN  = -0.01   # bottom of the cylinder (relative to plate centre Z)
    PLATE_Z_MAX  =  0.065   # top of the cylinder

    DEBUG_DRAW            = False  # set to True to visualize the grip axis in the Isaac Sim viewport
    ORANGE_HELD_MAX_DIST  = 0.06   # max tip-to-orange distance (m) to consider orange still held
    PLACE_GRIPPER_Z_MIN   = 0.04    # gripper tip must be at or above this env-relative Z to confirm place

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
        px, py, pz = self._plate_pos[0].item(), self._plate_pos[1].item(), self._plate_pos[2].item()
        ox, oy, oz = orange_pos[0].item(), orange_pos[1].item(), orange_pos[2].item()
        xy_dist = ((ox - px)**2 + (oy - py)**2) ** 0.5
        return xy_dist < self.PLATE_RADIUS and pz + self.PLATE_Z_MIN < oz < pz + self.PLATE_Z_MAX

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
        """Draw the cylindrical plate check volume in the Isaac Sim viewport."""
        if not self.DEBUG_DRAW or self._origin is None or self._plate_pos is None:
            return
        try:
            import omni.debugdraw
            import carb
            import math
            draw = omni.debugdraw.get_debug_draw_interface()
        except Exception:
            return

        p  = self._plate_pos + self._origin
        cx, cy, cz = p[0].item(), p[1].item(), p[2].item()
        z0 = cz + self.PLATE_Z_MIN
        z1 = cz + self.PLATE_Z_MAX
        r  = self.PLATE_RADIUS
        color, w, N = 0xFFFF8800, 2.0, 24  # orange, line width, segments

        pts_bot = [carb.Float3(cx + r * math.cos(2*math.pi*i/N),
                               cy + r * math.sin(2*math.pi*i/N), z0) for i in range(N)]
        pts_top = [carb.Float3(cx + r * math.cos(2*math.pi*i/N),
                               cy + r * math.sin(2*math.pi*i/N), z1) for i in range(N)]

        for i in range(N):
            j = (i + 1) % N
            draw.draw_line(pts_bot[i], color, w, pts_bot[j], color, w)  # bottom ring
            draw.draw_line(pts_top[i], color, w, pts_top[j], color, w)  # top ring
            draw.draw_line(pts_bot[i], color, w, pts_top[i], color, w)  # vertical edge

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

        px, py, pz   = plate_pos[0].item(), plate_pos[1].item(), plate_pos[2].item()
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
            ox, oy, oz = opos[0].item(), opos[1].item(), opos[2].item()
            xy_dist  = ((ox - px)**2 + (oy - py)**2) ** 0.5
            in_plate = (
                xy_dist < self.PLATE_RADIUS
                and pz + self.PLATE_Z_MIN < oz < pz + self.PLATE_Z_MAX
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
            ox, oy, oz = opos[0].item(), opos[1].item(), opos[2].item()
            xy_dist  = ((ox - px)**2 + (oy - py)**2) ** 0.5
            in_plate = (
                xy_dist < self.PLATE_RADIUS
                and pz + self.PLATE_Z_MIN < oz < pz + self.PLATE_Z_MAX
            )
            stable_frames, _ = self._stability.get(target, (0, None))
            held, held_dist  = self._is_orange_held(opos)
            gripper_z = (self._gripper_tip[2].item() - pz) if self._gripper_tip is not None else float("nan")
            r_sym  = "✓" if xy_dist < self.PLATE_RADIUS else "✗"
            z_sym  = "✓" if pz + self.PLATE_Z_MIN < oz < pz + self.PLATE_Z_MAX else "✗"
            s_sym  = "✓" if stable_frames >= self.stability_frames else "✗"
            d_sym  = "✓" if held else "✗"
            gz_sym = "✓" if gripper_z >= self.PLACE_GRIPPER_Z_MIN else "✗"
            g_sym  = "✓" if gripper_open else "✗"
            lines = [
                f"  🍊 PLACE [{display}]  step {step_count}",
                f"     Held:       {held_dist:.4f} < {self.ORANGE_HELD_MAX_DIST}  {d_sym}  (info)",
                f"     XY dist:    {xy_dist:.4f} < {self.PLATE_RADIUS}  {r_sym}",
                f"     Orange Z:   {oz - pz:.4f}  ∈ [{self.PLATE_Z_MIN}, {self.PLATE_Z_MAX}]  {z_sym}",
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
            idx     = joint_names.index(joint_name)
            val     = joint_deg[idx].item()
            ok      = "✓" if lo < val < hi else "✗"
            lines.append(f"     {joint_name:<14s} {val:+7.1f}°  ∈ [{lo:.0f}, {hi:.0f}]  {ok}")
        lines.append(f"     Patience:      {self._counter}/{self.patience_frames}")
        self._live_update(lines)

        if self._counter == self.patience_frames:
            self._fired        = True
            self._status_lines = 0
            print(f"  🏠 HOME CONFIRMED at step {step_count}\n")