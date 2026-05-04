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
        xy_threshold=0.04,          # max EE–orange XY distance to confirm grasp (m)
        grasp_z_min=0.03,           # min EE Z above orange centre to confirm grasp (m)
        grasp_threshold=0.60,       # gripper joint value: above = open, below = closed
        lift_height_threshold=0.1,  # min height gain from initial Z to confirm lift (m)
        orange_names=("Orange001", "Orange002", "Orange003"),
        stability_frames=10,        # frames orange must be stationary inside plate to confirm place
        stability_tolerance=0.001,  # max orange movement per frame to count as stationary (m)
    ):
        self.patience_frames       = patience_frames
        self.xy_threshold          = xy_threshold
        self.grasp_z_min           = grasp_z_min
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
        origin = env.scene.env_origins[0]
        ee_pos      = env.scene["ee_frame"].data.target_pos_w[0, 1, :] - origin
        gripper_pos = env.scene["robot"].data.joint_pos[0, -1].item()
        plate_pos   = env.scene["Plate"].data.root_pos_w[0] - origin
        orange_positions = {
            name: env.scene[name].data.root_pos_w[0] - origin
            for name in self.orange_names
        }
        return ee_pos, gripper_pos, plate_pos, orange_positions

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

    # ----------------------------------------------------------
    # Main entry point — call every step
    # ----------------------------------------------------------

    def check_status(self, env, step_count):
        """Run all subtask checks for the current step."""
        ee_pos, gripper_pos, plate_pos, orange_positions = self._get_env_data(env)

        if step_count == 0:
            for name, pos in orange_positions.items():
                self.initial_orange_z[name] = pos[2].item()

        self._check_grasp(ee_pos, gripper_pos, orange_positions, step_count)
        self._check_lift(ee_pos, gripper_pos, orange_positions, step_count)
        self._check_place(plate_pos, orange_positions, gripper_pos, step_count)

    # ----------------------------------------------------------
    # 1. Grasp check
    # ----------------------------------------------------------

    def _check_grasp(self, ee_pos, gripper_pos, orange_positions, step_count):
        """Live-display grasp conditions each step; confirm once patience is reached."""
        if self._grasp_confirmed:
            return

        gripper_closed = gripper_pos < self.grasp_threshold
        grasping = None

        if gripper_closed:
            for name, pos in orange_positions.items():
                if name in self.placed_oranges or name not in self.initial_orange_z:
                    continue
                dx = abs(ee_pos[0] - pos[0]).item()
                dy = abs(ee_pos[1] - pos[1]).item()
                xy_dist = (dx**2 + dy**2) ** 0.5
                dz = (ee_pos[2] - pos[2]).item()
                if xy_dist < self.xy_threshold and dz >= self.grasp_z_min:
                    grasping = (name, pos, xy_dist, dz)
                    break

        if grasping:
            self.grasp_counter += 1
        else:
            self.grasp_counter = 0

        g_sym = "✓" if gripper_closed else "✗"
        if grasping:
            name, pos, xy_dist, dz = grasping
            z_sym = "✓"
            lines = [
                f"  ✊ GRASP [{name}]  step {step_count}",
                f"     Gripper:  {gripper_pos:.4f} < {self.grasp_threshold} (closed)  {g_sym}",
                f"     XY dist:  {xy_dist:.4f} < {self.xy_threshold}  ✓",
                f"     Z offset: {dz:.4f} >= {self.grasp_z_min}  {z_sym}",
                f"     Patience: {self.grasp_counter}/{self.patience_frames}",
            ]
        else:
            lines = [
                f"  ✊ GRASP  step {step_count}",
                f"     Gripper:  {gripper_pos:.4f} < {self.grasp_threshold} (closed)  {g_sym}",
                f"     XY dist:  no candidate in range  ✗",
                f"     Patience: {self.grasp_counter}/{self.patience_frames}",
            ]
        self._live_update(lines)

        if self.grasp_counter == self.patience_frames:
            self._grasp_confirmed = True
            self._place_confirmed = False  # allow place check for new orange cycle
            self.active_orange    = grasping[0]
            self._status_lines    = 0
            print(f"  ✊ GRASP CONFIRMED: {self.active_orange} is now the active orange  (step {step_count})\n")
            self._pause()

    # ----------------------------------------------------------
    # 2. Lift check
    # ----------------------------------------------------------

    def _check_lift(self, ee_pos, gripper_pos, orange_positions, step_count):
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

    def __init__(self, patience_frames=5, joint_threshold=0.05):
        self.patience_frames  = patience_frames
        self.joint_threshold  = joint_threshold
        self._start_joint_pos = None
        self._counter         = 0
        self._fired           = False
        self._status_lines    = 0

    def reset(self, start_joint_pos: np.ndarray):
        self._start_joint_pos = start_joint_pos.copy()
        self._counter         = 0
        self._fired           = False
        self._status_lines    = 0

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
        if self._start_joint_pos is None or self._fired:
            return
        current = env.scene["robot"].data.joint_pos[0].cpu().numpy()
        max_dev = np.abs(current - self._start_joint_pos).max()

        if max_dev < self.joint_threshold:
            self._counter += 1
        else:
            self._counter = 0

        d_sym = "✓" if max_dev < self.joint_threshold else "✗"
        self._live_update([
            f"  🏠 HOME  step {step_count}",
            f"     Max joint dev: {max_dev:.4f} < {self.joint_threshold}  {d_sym}",
            f"     Patience:      {self._counter}/{self.patience_frames}",
        ])

        if self._counter == self.patience_frames:
            self._fired        = True
            self._status_lines = 0
            print(f"  🏠 HOME CONFIRMED at step {step_count}\n")