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

    # Plate bounds relative to plate centre (m). Shared by _check_place and count_oranges_in_plate.
    PLATE_X_RANGE    = (-0.10, 0.10)
    PLATE_Y_RANGE    = (-0.10, 0.10)
    PLATE_Z_RANGE    = (-0.01, 0.05)

    def __init__(
        self,
        block = False,              # if True, pause after each confirmed event
        patience_frames=10,         # consecutive frames required to confirm grasp or lift
        centering_threshold=0.03,   # max distance from orange centre to grip-axis segment (m)
        closure_threshold=0.065,    # max finger-gap to count as closed around orange (m)
        grip_t_min=0.3,             # min projection parameter: orange must not be outside the fingers
        grip_t_max=0.7,             # max projection parameter
        grasp_threshold=0.60,       # gripper joint value: above = open, below = closed (used by lift/place)
        lift_height_threshold=0.1,  # min height gain from initial Z to confirm lift (m)
        orange_names=("Orange001", "Orange002", "Orange003"),
        stability_frames=10,        # frames orange must be stationary inside plate to confirm place
        stability_tolerance=0.001,  # max orange movement per frame to count as stationary (m)
    ):
        self.patience_frames       = patience_frames
        self.centering_threshold   = centering_threshold
        self.closure_threshold     = closure_threshold
        self.grip_t_min            = grip_t_min
        self.grip_t_max            = grip_t_max
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
        gripper_tip = frames[frame_names.index("gripper_tip")] - self._origin
        jaw_tip     = frames[frame_names.index("jaw_tip")]     - self._origin
        gripper_pos = env.scene["robot"].data.joint_pos[0, -1].item()
        plate_pos   = env.scene["Plate"].data.root_pos_w[0] - self._origin
        orange_positions = {
            name: env.scene[name].data.root_com_pos_w[0] - self._origin
            for name in self.orange_names
        }
        return gripper_tip, jaw_tip, gripper_pos, plate_pos, orange_positions

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

    def _draw_grip_axis(self, gripper_tip, jaw_tip, best_orange_pos, best_proj, meets):
        """Draw the grip axis and centering debug geometry in the Isaac Sim viewport.

        Draws (in world space, cleared each call):
          - Grip axis segment gripper_tip → jaw_tip  (green = all conditions met, red otherwise)
          - Line from orange centre to its projection on the axis  (yellow)
          - Dot at the projection point  (yellow)
        """
        if self._origin is None:
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

            # Horizontal plane at the orange's Z height
            wp  = best_orange_pos + self._origin
            ox, oy, oz = wp[0].item(), wp[1].item(), wp[2].item()
            half, step = 0.10, 0.05
            n = round(half / step)
            plane_color = 0xFFFFFFFF  # white
            for i in range(-n, n + 1):
                y = oy + i * step
                draw.draw_line(carb.Float3(ox - half, y, oz), plane_color, 1.0,
                               carb.Float3(ox + half, y, oz), plane_color, 1.0)
            for i in range(-n, n + 1):
                x = ox + i * step
                draw.draw_line(carb.Float3(x, oy - half, oz), plane_color, 1.0,
                               carb.Float3(x, oy + half, oz), plane_color, 1.0)

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

    # ----------------------------------------------------------
    # Main entry point — call every step
    # ----------------------------------------------------------

    def check_status(self, env, step_count):
        """Run all subtask checks for the current step."""
        gripper_tip, jaw_tip, gripper_pos, plate_pos, orange_positions = self._get_env_data(env)

        if step_count == 0:
            for name, pos in orange_positions.items():
                self.initial_orange_z[name] = pos[2].item()

        self._check_grasp(gripper_tip, jaw_tip, orange_positions, step_count)
        self._check_lift(gripper_pos, orange_positions, step_count)
        self._check_place(plate_pos, orange_positions, gripper_pos, step_count)

    # ----------------------------------------------------------
    # 1. Grasp check
    # ----------------------------------------------------------

    def _check_grasp(self, gripper_tip, jaw_tip, orange_positions, step_count):
        """Live-display grasp conditions each step; confirm once patience is reached.

        Centering: distance from orange centre to the gripper_tip→jaw_tip segment.
        Closure:   distance between the two fingertips (finger gap).
        Both must be under threshold for patience_frames consecutive steps.
        Always tracks the closest unplaced orange to show live feedback before active_orange is set.
        """
        if self._grasp_confirmed:
            return

        axis    = jaw_tip - gripper_tip
        axis_sq = torch.dot(axis, axis).item()
        gap     = axis_sq ** 0.5

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
                 and t_ok)

        if meets:
            self.grasp_counter += 1
        else:
            self.grasp_counter = 0

        # Live display — always show best candidate
        c_sym = "✓" if best_dist < self.centering_threshold else "✗"
        g_sym = "✓" if gap < self.closure_threshold else "✗"
        t_sym = "✓" if t_ok else "✗"
        label = best_name if best_name else "?"
        lines = [
            f"  ✊ GRASP [{label}]  step {step_count}",
            f"     Centering: {best_dist:.4f} < {self.centering_threshold}  {c_sym}",
            f"     Closure:   {gap:.4f} < {self.closure_threshold}  {g_sym}",
            f"     Grip pos:  t={best_t:.2f}  ∈ [{self.grip_t_min}, {self.grip_t_max}]  {t_sym}",
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
            h_sym       = "✓" if height_gain > self.lift_height_threshold else "✗"
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
            in_plate = (
                (px + self.PLATE_X_RANGE[0] < ox < px + self.PLATE_X_RANGE[1])
                and (py + self.PLATE_Y_RANGE[0] < oy < py + self.PLATE_Y_RANGE[1])
                and (pz + self.PLATE_Z_RANGE[0] < oz < pz + self.PLATE_Z_RANGE[1])
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
            if stable_frames >= self.stability_frames and gripper_open:
                newly_confirmed.append((name, opos, stable_frames))
                self.placed_oranges.add(name)

        # Live status
        display = target if target else "?"
        g_sym   = "✓" if gripper_open else "✗"
        if target and target in orange_positions:
            opos  = orange_positions[target]
            ox, oy, oz = opos[0].item(), opos[1].item(), opos[2].item()
            in_plate = (
                (px + self.PLATE_X_RANGE[0] < ox < px + self.PLATE_X_RANGE[1])
                and (py + self.PLATE_Y_RANGE[0] < oy < py + self.PLATE_Y_RANGE[1])
                and (pz + self.PLATE_Z_RANGE[0] < oz < pz + self.PLATE_Z_RANGE[1])
            )
            stable_frames, _ = self._stability.get(target, (0, None))
            p_sym = "✓" if in_plate else "✗"
            s_sym = "✓" if stable_frames >= self.stability_frames else "✗"
            lines = [
                f"  🍊 PLACE [{display}]  step {step_count}",
                f"     In plate:  {p_sym}",
                f"     Stable:    {stable_frames}/{self.stability_frames}  {s_sym}",
                f"     Gripper:   {gripper_pos:.4f} > {self.grasp_threshold} (open)  {g_sym}",
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