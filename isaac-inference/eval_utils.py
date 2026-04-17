"""
Evaluation utilities for robotic pick-and-place tasks.

Provides:
  - save_positions / count_oranges_in_plate : geometric scene-state helpers
  - save_camera_snapshots                   : debug image dumps
  - EvaluationTracker                       : per-episode logging, tqdm bar, final summary + file export
  - SubtaskTracker                          : fine-grained phase tracker (multi-object pick-and-place)
"""

import os
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
    """Detects three independent robot events each step and prints a debug report when each fires.

    Events detected:
      1. Hover  — EE has been positioned above an unplaced orange for N consecutive frames
                  while the gripper is open.
      2. Lift   — Gripper is closed and an unplaced orange has been raised above its
                  initial height for N consecutive frames.
      3. Place  — An orange has been stably inside the plate for N consecutive frames
                  while the gripper is open (i.e. it has been released).

    All three checks run independently every step. Once an orange is confirmed as placed
    it is excluded from the hover and lift checks.
    Each event fires a print + pause exactly once per occurrence.
    """

    # Plate bounds relative to plate centre (m). Shared by _check_place and count_oranges_in_plate.
    PLATE_X_RANGE    = (-0.10, 0.10)
    PLATE_Y_RANGE    = (-0.10, 0.10)
    PLATE_Z_RANGE    = (-0.01, 0.05)

    def __init__(
        self,
        block = False,              # if True, skip all checks and print a single message at init
        patience_frames=5,        # frames required to confirm hover or lift
        xy_threshold=0.04,        # max EE–orange XY distance to count as hovering (m)
        z_offset_min=0.0,         # min EE Z above orange to count as hovering (m)
        z_offset_max=0.05,        # max EE Z above orange to count as hovering (m)
        grasp_threshold=0.60,     # gripper joint value: above = open, below = closed
        lift_height_threshold=0.1,# min height gain from initial Z to count as lifted (m)
        orange_names=("Orange001", "Orange002", "Orange003"),
        stability_frames=10,      # frames orange must be stationary inside plate to confirm place
        stability_tolerance=0.001,# max orange movement per frame to count as stationary (m)
    ):
        self.patience_frames       = patience_frames
        self.xy_threshold          = xy_threshold
        self.z_offset_min          = z_offset_min
        self.z_offset_max          = z_offset_max
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
        self.placed_oranges    = set()   # names of oranges confirmed placed in plate
        self.initial_orange_z  = {}      # initial Z height per orange, recorded at step 0
        self.hover_counter     = 0       # consecutive frames EE has been hovering over an orange
        self.lift_counter      = 0       # consecutive frames an orange has been lifted
        # Maps orange_name -> (consecutive_stable_frames, last_position_tensor | None)
        self._stability: dict[str, tuple[int, torch.Tensor | None]] = {}

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

    # ----------------------------------------------------------
    # Main entry point — call every step
    # ----------------------------------------------------------

    def check_status(self, env, step_count):
        """Run all three subtask checks for the current step."""
        ee_pos, gripper_pos, plate_pos, orange_positions = self._get_env_data(env)

        # Record initial orange heights on the first step
        if step_count == 0:
            for name, pos in orange_positions.items():
                self.initial_orange_z[name] = pos[2].item()

        self._check_hover(ee_pos, gripper_pos, orange_positions, step_count)
        self._check_lift(ee_pos, gripper_pos, orange_positions, step_count)
        self._check_place(plate_pos, orange_positions, gripper_pos, step_count)

    # ----------------------------------------------------------
    # 1. Hover check
    # ----------------------------------------------------------

    def _check_hover(self, ee_pos, gripper_pos, orange_positions, step_count):
        """Detect when the EE is positioned above an unplaced orange, ready to grasp.

        Increments hover_counter while all conditions hold; resets it otherwise.
        Fires (print + pause) on the frame the counter reaches patience_frames.
        """
        gripper_open = gripper_pos > self.grasp_threshold
        hovering_over = None

        if gripper_open:
            for name, pos in orange_positions.items():
                if name in self.placed_oranges:
                    continue

                dx = abs(ee_pos[0] - pos[0]).item()
                dy = abs(ee_pos[1] - pos[1]).item()
                dz = (ee_pos[2] - pos[2]).item()
                xy_dist  = (dx**2 + dy**2) ** 0.5
                on_ground = abs(pos[2].item() - self.initial_orange_z.get(name, pos[2].item())) < 0.01

                if (xy_dist < self.xy_threshold
                        and self.z_offset_min < dz < self.z_offset_max
                        and on_ground):
                    hovering_over = (name, pos, xy_dist, dz, on_ground)
                    break

        if hovering_over:
            self.hover_counter += 1
        else:
            self.hover_counter = 0

        if self.hover_counter == self.patience_frames:
            name, pos, xy_dist, dz, on_ground = hovering_over
            orange_z      = pos[2].item()
            initial_z     = self.initial_orange_z.get(name, orange_z)
            z_delta       = abs(orange_z - initial_z)
            print(
                f"\n  👁  HOVER: EE over {name} at step {step_count}\n"
                f"  ── Raw values ──\n"
                f"     EE pos:      ({ee_pos[0].item():.4f}, {ee_pos[1].item():.4f}, {ee_pos[2].item():.4f})\n"
                f"     Orange pos:  ({pos[0].item():.4f}, {pos[1].item():.4f}, {pos[2].item():.4f})\n"
                f"     Gripper pos: {gripper_pos:.4f}\n"
                f"  ── Conditions ──\n"
                f"     XY dist:    {xy_dist:.4f} < {self.xy_threshold} ✓\n"
                f"     Z offset:   {dz:.4f} in ({self.z_offset_min}, {self.z_offset_max}) ✓\n"
                f"     On ground:  |{orange_z:.4f} - {initial_z:.4f}| = {z_delta:.4f} < 0.02 ✓\n"
                f"     Gripper:    {gripper_pos:.4f} > {self.grasp_threshold} (open) ✓\n"
                f"     Patience:   {self.patience_frames}/{self.patience_frames} ✓"
            )
            self._pause()

    # ----------------------------------------------------------
    # 2. Lift check
    # ----------------------------------------------------------

    def _check_lift(self, ee_pos, gripper_pos, orange_positions, step_count):
        """Detect when the robot has successfully grasped and lifted an orange.

        Requires the gripper to be closed and an unplaced orange to be well above
        its initial resting height for patience_frames consecutive frames.
        """
        gripper_closed  = gripper_pos < self.grasp_threshold
        currently_lifted = None

        if gripper_closed:
            for name, pos in orange_positions.items():
                if name in self.placed_oranges or name not in self.initial_orange_z:
                    continue
                if pos[2].item() > self.initial_orange_z[name] + self.lift_height_threshold:
                    currently_lifted = (name, pos, pos[2].item(), self.initial_orange_z[name])
                    break

        if currently_lifted:
            self.lift_counter += 1
        else:
            self.lift_counter = 0

        if self.lift_counter == self.patience_frames:
            name, pos, current_z, initial_z = currently_lifted
            print(
                f"\n  🤏 LIFT: {name} lifted at step {step_count}\n"
                f"  ── Raw values ──\n"
                f"     EE pos:      ({ee_pos[0].item():.4f}, {ee_pos[1].item():.4f}, {ee_pos[2].item():.4f})\n"
                f"     Orange pos:  ({pos[0].item():.4f}, {pos[1].item():.4f}, {pos[2].item():.4f})\n"
                f"     Gripper pos: {gripper_pos:.4f}\n"
                f"     Initial Z:   {initial_z:.4f}\n"
                f"  ── Conditions ──\n"
                f"     Gripper:  {gripper_pos:.4f} < {self.grasp_threshold} (closed) ✓\n"
                f"     Lifted:   {current_z:.4f} > {initial_z + self.lift_height_threshold:.4f} ✓\n"
                f"     Patience: {self.patience_frames}/{self.patience_frames} ✓"
            )
            self._pause()

    # ----------------------------------------------------------
    # 3. Place check
    # ----------------------------------------------------------

    def _check_place(self, plate_pos, orange_positions, gripper_pos, step_count):
        """Detect when an orange has been released and is resting stably inside the plate.

        For each unplaced orange currently within the plate bounds, we track how many
        consecutive frames it has remained stationary. Once it reaches stability_frames
        AND the gripper is open (confirming the orange has been released), the orange
        is marked as placed.
        """
        px, py, pz   = plate_pos[0].item(), plate_pos[1].item(), plate_pos[2].item()
        gripper_open = gripper_pos > self.grasp_threshold
        newly_confirmed = []

        for name, opos in orange_positions.items():
            if name in self.placed_oranges:
                continue

            ox, oy, oz = opos[0].item(), opos[1].item(), opos[2].item()
            in_plate = (
                (px + self.PLATE_X_RANGE[0] < ox < px + self.PLATE_X_RANGE[1])
                and (py + self.PLATE_Y_RANGE[0] < oy < py + self.PLATE_Y_RANGE[1])
                and (pz + self.PLATE_Z_RANGE[0] < oz < pz + self.PLATE_Z_RANGE[1])
            )

            if not in_plate:
                # Reset stability counter when orange leaves the plate bounds
                self._stability[name] = (0, None)
                continue

            # Update stability counter: reset if the orange moved, otherwise increment
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

        if not newly_confirmed:
            return

        current_count   = len(self.placed_oranges)
        confirmed_names = ", ".join(n for n, _, _ in newly_confirmed)
        stable_counts   = ", ".join(str(f) for _, _, f in newly_confirmed)
        orange_lines    = [
            f"     {name}: ({opos[0].item():.4f}, {opos[1].item():.4f}, {opos[2].item():.4f})"
            + (" ✓ just placed"    if any(n == name for n, _, _ in newly_confirmed) else
               " ✓ already placed" if name in self.placed_oranges else "")
            for name, opos in orange_positions.items()
        ]
        print(
            f"\n  🍊 PLACE: {confirmed_names} confirmed in plate at step {step_count} "
            f"({current_count}/{self.total_oranges} total)\n"
            f"  ── Raw values ──\n"
            f"     Plate pos: ({plate_pos[0].item():.4f}, {plate_pos[1].item():.4f}, {plate_pos[2].item():.4f})\n"
            + "\n".join(orange_lines) + "\n"
            f"  ── Conditions ──\n"
            f"     Stable frames: {stable_counts} ≥ {self.stability_frames} ✓\n"
            f"     Stability tol: ≤ {self.stability_tolerance} m/frame ✓\n"
            f"     Gripper:       {gripper_pos:.4f} > {self.grasp_threshold} (open) ✓"
        )
        self._pause()