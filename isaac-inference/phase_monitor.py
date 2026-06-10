import os
import sys

import numpy as np
import torch

from eval_utils import SubtaskTracker, classify_orange_positions, is_plate_upside_down, plate_position_metrics


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "2"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


class PhaseMonitor:
    """Live passive subtask monitor for whole-episode policies.

    Flat policies do not expose intent. This monitor records an observer-inferred
    trace from physics state only: when not lifting or placing, it treats the
    behavior as a grasp/search attempt toward the nearest stable unplaced orange.
    """

    TRACE_SOURCE = "flat_observed_physics"
    SCHEMA_VERSION = 2

    def __init__(self, model_id=None):
        self.model_id = model_id
        self.tracker = SubtaskTracker(patience_frames=10)
        self.live_enabled = _env_flag("PHASE_DEBUG_LIVE", True)
        self.live_every_steps = max(1, _env_int("PHASE_DEBUG_EVERY_STEPS", 5))
        self.retarget_frames = max(1, _env_int("PHASE_DEBUG_RETARGET_FRAMES", 10))
        self.bounce_frames = max(1, _env_int("PHASE_DEBUG_BOUNCE_FRAMES", 5))
        self.reset()

    def reset(self):
        self.current_phase = "SEARCHING"
        self.active_orange = None
        self.initial_orange_z = {}
        self.initial_labels = {}
        self.initial_scene = None
        self.last_plate_pos = None
        self.last_orange_positions = {}
        self.placed_oranges = set()
        self.placed_outside_streak = {}
        self.opportunistic_place_stability = {}

        self.grasp_streak = 0
        self.lift_streak = 0
        self.place_stability = {}
        self.inferred_grasp_target = None
        self.pending_grasp_target = None
        self.pending_grasp_streak = 0

        self.timeline = []
        self.subtask_attempts = []
        self.events = []
        self._active_attempt = None
        self._attempt_id = 0
        self._live_active = False

    def warm_up(self, env):
        _, _, _, _, _, plate_pos, orange_positions = self.tracker._get_env_data(env)
        self.initial_orange_z = {
            name: pos[2].item()
            for name, pos in orange_positions.items()
        }
        self.initial_labels = classify_orange_positions(orange_positions)
        self.initial_scene = self._scene(plate_pos, orange_positions)
        self.last_plate_pos = plate_pos
        self.last_orange_positions = {name: pos.clone() for name, pos in orange_positions.items()}
        self._add_event(0, "episode_started", phase="OBSERVE", outcome="started")

    def update(self, env, step_count, episode):
        t = self.tracker
        gripper_tip, jaw_tip, gripper_pos, gf, jf, plate_pos, orange_positions = t._get_env_data(env)
        self.last_plate_pos = plate_pos
        self.last_orange_positions = {name: pos.clone() for name, pos in orange_positions.items()}

        self._check_placed_bounce(episode, step_count, plate_pos, orange_positions)
        if self.current_phase == "SEARCHING":
            self._update_grasp(episode, step_count, gripper_tip, jaw_tip, orange_positions, gf, jf)
        elif self.current_phase == "LIFTING":
            self._update_lift(episode, step_count, gripper_pos, orange_positions)
        elif self.current_phase == "PLACING":
            self._update_place(episode, step_count, plate_pos, orange_positions, gripper_pos)

    def finish_line(self):
        if self._live_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._live_active = False

    def build_record(
        self,
        episode,
        step_count,
        oranges_in_plate,
        *,
        end_reason,
        is_success,
        final_positions=None,
    ):
        self._finish_active_for_episode_end(step_count, end_reason)
        final_scene = self._final_scene(final_positions)
        self._add_event(
            step_count,
            "episode_finished",
            outcome="success" if is_success else "failure",
            reason=end_reason,
            oranges_in_plate=int(oranges_in_plate),
        )
        return {
            "episode_summary": {
                "episode": int(episode),
                "model_id": self.model_id,
                "total_steps": int(step_count),
                "final_oranges_in_plate": int(oranges_in_plate),
                "end_reason": end_reason,
                "success": bool(is_success),
                "trace_source": self.TRACE_SOURCE,
                "inferred": True,
            },
            "initial_scene": self.initial_scene,
            "final_scene": final_scene,
            "timeline": self.timeline,
            "subtask_attempts": self.subtask_attempts,
            "phase_debug": {
                "schema_version": self.SCHEMA_VERSION,
                "trace_source": self.TRACE_SOURCE,
                "inferred": True,
                "heuristic": "non_lift_non_place_means_inferred_grasp_search",
                "retarget_frames": self.retarget_frames,
                "final_phase": self.current_phase,
                "active_orange": self.active_orange,
                "placed_oranges_observed": sorted(self.placed_oranges),
                "events": self.events,
                "summary": self._debug_summary(),
            },
        }

    def _update_grasp(self, episode, step_count, gripper_tip, jaw_tip, orange_positions, gf, jf):
        candidate = self._grasp_candidate(gripper_tip, jaw_tip, orange_positions, gf, jf)
        if candidate is None:
            self._live(episode, step_count, "SEARCHING", "no unplaced orange candidate")
            return

        self._update_stable_grasp_target(step_count, candidate)
        if self.inferred_grasp_target is None:
            return

        if candidate["ready"]:
            if candidate["name"] != self.inferred_grasp_target:
                self.grasp_streak = 0
            else:
                self.grasp_streak += 1
        else:
            self.grasp_streak = 0

        self._live(
            episode,
            step_count,
            "GRASP?",
            f"inferred={self.inferred_grasp_target} nearest={candidate['name']} "
            f"center={candidate['center_dist']:.3f}m gap={candidate['gap']:.3f}m "
            f"force={candidate['min_force']:.1f}N",
        )

        if self.grasp_streak >= self.tracker.patience_frames:
            actual = candidate["name"]
            self.active_orange = actual
            self.current_phase = "LIFTING"
            self.lift_streak = 0
            self._finish_attempt(
                step_count,
                result="success",
                actual_orange=actual,
                metrics=self._grasp_metrics(candidate),
            )
            self._event(
                episode,
                step_count,
                "GRASP",
                "grasp_success",
                actual_orange=actual,
                metrics=self._grasp_metrics(candidate),
            )
            self._start_attempt(step_count, "LIFT", actual, metrics={"height_gain_m": 0.0})

    def _update_lift(self, episode, step_count, gripper_pos, orange_positions):
        name = self.active_orange
        if name not in orange_positions:
            self._fail_active_and_reset(episode, step_count, "LIFT", "episode_ended", None)
            return

        opos = orange_positions[name]
        held, held_dist = self.tracker._is_orange_held(opos)
        initial_z = self.initial_orange_z.get(name, opos[2].item())
        height_gain = opos[2].item() - initial_z
        gripper_closed = gripper_pos < self.tracker.grasp_threshold
        metrics = {
            "height_gain_m": round(float(height_gain), 5),
            "held": bool(held),
            "tip_distance_m": round(float(held_dist), 5),
            "gripper_closed": bool(gripper_closed),
            "gripper_position": round(float(gripper_pos), 5),
        }

        if not held:
            self._fail_active_and_reset(episode, step_count, "LIFT", "dropped_during_lift", name, metrics)
            return

        place_metrics = self._observe_place_metrics(name, opos, self.last_plate_pos, gripper_pos)
        if self._place_confirmed(place_metrics):
            self.placed_oranges.add(name)
            self.placed_outside_streak.pop(name, None)
            self._finish_attempt(
                step_count,
                result="skipped",
                actual_orange=name,
                failure_reason="inferred_place_without_lift",
                metrics={**metrics, "place_confirmed_before_lift": True},
            )
            self._event(
                episode,
                step_count,
                "LIFT",
                "lift_skipped",
                actual_orange=name,
                outcome="skipped",
                reason="inferred_place_without_lift",
                metrics={**metrics, "place_confirmed_before_lift": True},
            )
            self._start_attempt(
                step_count,
                "PLACE",
                name,
                metrics={"inferred_place_without_lift": True},
            )
            self._finish_attempt(
                step_count,
                result="success",
                actual_orange=name,
                metrics={**place_metrics, "placed_count": len(self.placed_oranges), "inferred_place_without_lift": True},
            )
            self._event(
                episode,
                step_count,
                "PLACE",
                "place_success",
                actual_orange=name,
                metrics={**place_metrics, "placed_count": len(self.placed_oranges), "inferred_place_without_lift": True},
            )
            self._reset_to_search()
            return

        lifted = gripper_closed and height_gain > self.tracker.lift_height_threshold
        self.lift_streak = self.lift_streak + 1 if lifted else 0
        grip = "closed" if gripper_closed else "open"
        self._live(
            episode,
            step_count,
            "LIFTING",
            f"target={name} height_gain={height_gain:.3f}m held=yes "
            f"tip_dist={held_dist:.3f}m gripper={grip}",
        )

        if self.lift_streak >= self.tracker.patience_frames:
            self.current_phase = "PLACING"
            self.place_stability.pop(name, None)
            self._finish_attempt(step_count, result="success", actual_orange=name, metrics=metrics)
            self._event(episode, step_count, "LIFT", "lift_success", actual_orange=name, metrics=metrics)
            self._start_attempt(step_count, "PLACE", name, metrics=metrics)

    def _update_place(self, episode, step_count, plate_pos, orange_positions, gripper_pos):
        name = self.active_orange
        if name not in orange_positions:
            self._fail_active_and_reset(episode, step_count, "PLACE", "episode_ended", None)
            return

        opos = orange_positions[name]
        metrics = self._observe_place_metrics(name, opos, plate_pos, gripper_pos, stability_store=self.place_stability)
        grip = "open" if metrics["gripper_open"] else "closed"
        state = "inside_plate" if metrics["in_plate"] else "outside_plate"
        self._live(
            episode,
            step_count,
            "PLACING",
            f"target={name} {state} held={'yes' if metrics['held'] else 'no'} "
            f"tip_dist={metrics['tip_distance_m']:.3f}m gripper={grip} stable={metrics['stable_frames']}f",
        )

        if not metrics["held"] and not metrics["in_plate"]:
            self._fail_active_and_reset(episode, step_count, "PLACE", "dropped_during_place", name, metrics)
            return

        if self._place_confirmed(metrics):
            self.placed_oranges.add(name)
            self.placed_outside_streak.pop(name, None)
            self._finish_attempt(step_count, result="success", actual_orange=name, metrics=metrics)
            self._event(
                episode,
                step_count,
                "PLACE",
                "place_success",
                actual_orange=name,
                metrics={**metrics, "placed_count": len(self.placed_oranges)},
            )
            self._reset_to_search()

    def _update_stable_grasp_target(self, step_count, candidate):
        name = candidate["name"]
        if self.inferred_grasp_target is None:
            self.inferred_grasp_target = name
            self.pending_grasp_target = None
            self.pending_grasp_streak = 0
            self._start_attempt(step_count, "GRASP", name, metrics=self._grasp_metrics(candidate))
            return

        if name == self.inferred_grasp_target:
            self.pending_grasp_target = None
            self.pending_grasp_streak = 0
            return

        if name == self.pending_grasp_target:
            self.pending_grasp_streak += 1
        else:
            self.pending_grasp_target = name
            self.pending_grasp_streak = 1

        if self.pending_grasp_streak >= self.retarget_frames:
            self._finish_attempt(
                step_count,
                result="retargeted",
                failure_reason="inferred_target_changed",
                metrics={"new_inferred_target_orange": name},
            )
            self._add_event(
                step_count,
                "grasp_retargeted",
                phase="GRASP",
                inferred_target_orange=name,
                inferred_target_label=self._label(name),
                outcome="retargeted",
                reason="inferred_target_changed",
            )
            self.inferred_grasp_target = name
            self.pending_grasp_target = None
            self.pending_grasp_streak = 0
            self.grasp_streak = 0
            self._start_attempt(step_count, "GRASP", name, metrics=self._grasp_metrics(candidate))

    def _check_placed_bounce(self, episode, step_count, plate_pos, orange_positions):
        for name in list(self.placed_oranges):
            if name not in orange_positions:
                continue
            opos = orange_positions[name]
            held, held_dist = self.tracker._is_orange_held(opos)
            xy_dist, z_offset, in_plate = self._plate_metrics(plate_pos, opos)
            if in_plate or held:
                self.placed_outside_streak[name] = 0
                continue
            streak = self.placed_outside_streak.get(name, 0) + 1
            self.placed_outside_streak[name] = streak
            if streak >= self.bounce_frames:
                self.placed_oranges.remove(name)
                self.placed_outside_streak.pop(name, None)
                metrics = {
                    "xy_distance_m": round(float(xy_dist), 5),
                    "z_offset_m": round(float(z_offset), 5),
                    "tip_distance_m": round(float(held_dist), 5),
                }
                self._event(
                    episode,
                    step_count,
                    "PLACE",
                    "placed_orange_left_plate",
                    actual_orange=name,
                    outcome="failure",
                    reason="placed_orange_left_plate",
                    metrics=metrics,
                )

    def _fail_active_and_reset(self, episode, step_count, phase, reason, actual_orange, metrics=None):
        self._finish_attempt(
            step_count,
            result="failure",
            actual_orange=actual_orange,
            failure_reason=reason,
            metrics=metrics,
        )
        self._event(
            episode,
            step_count,
            phase,
            f"{phase.lower()}_failure",
            actual_orange=actual_orange,
            outcome="failure",
            reason=reason,
            metrics=metrics,
        )
        self._reset_to_search()

    def _finish_active_for_episode_end(self, step_count, end_reason):
        if self._active_attempt is None:
            if not self.subtask_attempts:
                self._add_event(
                    step_count,
                    "no_confirmed_progress",
                    phase="GRASP",
                    outcome="failure",
                    reason="no_confirmed_progress",
                )
            return

        reason = "plate_flipped" if end_reason == "plate_flipped" else "episode_ended"
        has_confirmed_progress = any(
            event.get("event_type") in {"grasp_success", "lift_success", "place_success"}
            for event in self.events
        )
        if self._active_attempt["subtask"] == "GRASP" and not has_confirmed_progress:
            reason = "no_confirmed_progress"
        self._finish_attempt(step_count, result="failure", failure_reason=reason)
        finished = self.subtask_attempts[-1]
        self._add_event(
            step_count,
            f"{finished['subtask'].lower()}_failure",
            phase=finished["subtask"],
            inferred_target_orange=finished.get("inferred_target_orange"),
            inferred_target_label=finished.get("inferred_target_label"),
            outcome="failure",
            reason=reason,
        )

    def _start_attempt(self, step, subtask, inferred_target_orange, metrics=None):
        if self._active_attempt is not None:
            self._finish_attempt(
                step,
                result="failure",
                failure_reason="interrupted_by_new_attempt",
            )
        self._attempt_id += 1
        label = self._label(inferred_target_orange)
        self._active_attempt = {
            "attempt_id": self._attempt_id,
            "subtask": subtask,
            "start_step": int(step),
            "end_step": None,
            "duration_steps": None,
            "requested_orange": None,
            "requested_label": None,
            "inferred_target_orange": inferred_target_orange,
            "inferred_target_label": label,
            "actual_orange": None,
            "actual_label": None,
            "target_match": None,
            "result": None,
            "failure_reason": None,
            "metrics": metrics or {},
        }
        self._add_event(
            step,
            f"{subtask.lower()}_started",
            phase=subtask,
            inferred_target_orange=inferred_target_orange,
            inferred_target_label=label,
            outcome="started",
            metrics=metrics,
        )

    def _finish_attempt(self, step, result, actual_orange=None, failure_reason=None, metrics=None):
        if self._active_attempt is None:
            return
        attempt = self._active_attempt
        if actual_orange is not None:
            attempt["actual_orange"] = actual_orange
            attempt["actual_label"] = self._label(actual_orange)
        attempt["end_step"] = int(step)
        attempt["duration_steps"] = int(step) - attempt["start_step"]
        attempt["result"] = result
        attempt["failure_reason"] = failure_reason
        if metrics:
            attempt["metrics"] = {**attempt.get("metrics", {}), **metrics}
        self.subtask_attempts.append(attempt)
        self._active_attempt = None

    def _event(
        self,
        episode,
        step,
        phase,
        event_type,
        *,
        actual_orange=None,
        outcome="success",
        reason=None,
        metrics=None,
    ):
        self._add_event(
            step,
            event_type,
            phase=phase,
            actual_orange=actual_orange,
            actual_label=self._label(actual_orange),
            outcome=outcome,
            reason=reason,
            metrics=metrics,
        )
        detail = event_type if reason is None else f"{event_type} reason={reason}"
        if actual_orange:
            detail = f"{detail} target={actual_orange}"
        self._print_event(episode, step, phase if outcome == "success" else "FAIL", detail)

    def _add_event(self, step, event_type, phase=None, actual_orange=None,
                   actual_label=None, outcome=None, reason=None,
                   inferred_target_orange=None, inferred_target_label=None,
                   metrics=None, **details):
        event = {
            "step": int(step),
            "event_type": event_type,
            "phase": phase,
            "requested_orange": None,
            "requested_label": None,
            "inferred_target_orange": inferred_target_orange,
            "inferred_target_label": inferred_target_label,
            "actual_orange": actual_orange,
            "actual_label": actual_label,
            "outcome": outcome,
            "reason": reason,
        }
        if metrics:
            event["metrics"] = metrics
        if details:
            event["details"] = details
        self.timeline.append(event)
        if event_type != "episode_started":
            self.events.append(event)

    def _debug_summary(self):
        counts = {}
        failures = {}
        for attempt in self.subtask_attempts:
            subtask = attempt["subtask"]
            counts[subtask] = counts.get(subtask, 0) + 1
            reason = attempt.get("failure_reason")
            if reason:
                failures[reason] = failures.get(reason, 0) + 1
        return {
            "attempt_counts": counts,
            "failure_counts": failures,
            "event_count": len(self.events),
        }

    def _grasp_candidate(self, gripper_tip, jaw_tip, orange_positions, gf, jf):
        axis = jaw_tip - gripper_tip
        axis_sq = torch.dot(axis, axis).item()
        gap = axis_sq ** 0.5
        axis_unit = axis / (gap + 1e-8)
        gripper_force = abs(torch.dot(gf.to(axis_unit.device), axis_unit).item())
        jaw_force = abs(torch.dot(jf.to(axis_unit.device), axis_unit).item())

        best = None
        for name, orange_pos in orange_positions.items():
            if name in self.placed_oranges:
                continue
            t_raw = torch.dot(orange_pos - gripper_tip, axis).item() / (axis_sq + 1e-8)
            t_clamped = max(0.0, min(1.0, t_raw))
            proj = gripper_tip + t_clamped * axis
            center_dist = (orange_pos - proj).norm().item()
            if best is None or center_dist < best["center_dist"]:
                best = {
                    "name": name,
                    "center_dist": center_dist,
                    "gap": gap,
                    "t": t_raw,
                    "gripper_force": gripper_force,
                    "jaw_force": jaw_force,
                    "min_force": min(gripper_force, jaw_force),
                }

        if best is None:
            return None

        best["ready"] = (
            best["center_dist"] < self.tracker.centering_threshold
            and best["gap"] < self.tracker.closure_threshold
            and self.tracker.grip_t_min <= best["t"] <= self.tracker.grip_t_max
            and best["gripper_force"] >= self.tracker.contact_force_min
            and best["jaw_force"] >= self.tracker.contact_force_min
        )
        return best

    def _grasp_metrics(self, candidate):
        return {
            "center_distance_m": round(float(candidate["center_dist"]), 5),
            "finger_gap_m": round(float(candidate["gap"]), 5),
            "grip_axis_t": round(float(candidate["t"]), 5),
            "gripper_force_n": round(float(candidate["gripper_force"]), 5),
            "jaw_force_n": round(float(candidate["jaw_force"]), 5),
        }

    def _plate_metrics(self, plate_pos, orange_pos):
        return plate_position_metrics(
            plate_pos,
            orange_pos,
            plate_radius=self.tracker.PLATE_RADIUS,
            plate_z_min=self.tracker.PLATE_Z_MIN,
            plate_z_max=self.tracker.PLATE_Z_MAX,
            plate_quat=self.tracker._plate_quat,
            inner_radius=self.tracker.PLATE_INNER_RADIUS,
            cone_height=self.tracker.PLATE_CONE_HEIGHT,
        )

    def _observe_place_metrics(self, name, opos, plate_pos, gripper_pos, stability_store=None):
        stability_store = stability_store if stability_store is not None else self.opportunistic_place_stability
        xy_dist, z_offset, in_plate = self._plate_metrics(plate_pos, opos)
        prev_frames, prev_pos = stability_store.get(name, (0, None))
        moved = prev_pos is not None and (opos - prev_pos).norm().item() > self.tracker.stability_tolerance
        stable_frames = 0 if moved or not in_plate else prev_frames + 1
        stability_store[name] = (stable_frames, opos.clone())
        held, held_dist = self.tracker._is_orange_held(opos)
        gripper_open = gripper_pos > self.tracker.grasp_threshold
        gripper_z = self.tracker._gripper_tip[2].item() - plate_pos[2].item()
        return {
            "held": bool(held),
            "tip_distance_m": round(float(held_dist), 5),
            "xy_distance_m": round(float(xy_dist), 5),
            "z_offset_m": round(float(z_offset), 5),
            "in_plate": bool(in_plate),
            "stable_frames": int(stable_frames),
            "gripper_open": bool(gripper_open),
            "gripper_position": round(float(gripper_pos), 5),
            "gripper_z_above_plate_m": round(float(gripper_z), 5),
        }

    def _place_confirmed(self, metrics):
        return (
            metrics["stable_frames"] >= self.tracker.stability_frames
            and metrics["gripper_open"]
            and metrics["gripper_z_above_plate_m"] >= self.tracker.PLACE_GRIPPER_Z_MIN
        )

    def _reset_to_search(self):
        self.current_phase = "SEARCHING"
        self.active_orange = None
        self.inferred_grasp_target = None
        self.pending_grasp_target = None
        self.pending_grasp_streak = 0
        self.grasp_streak = 0
        self.lift_streak = 0
        self.place_stability = {}
        self.opportunistic_place_stability = {}

    def _final_scene(self, final_positions):
        if final_positions:
            plate_pos = final_positions.get("plate")
            orange_positions = {
                name: final_positions[name]
                for name in self.tracker.orange_names
                if name in final_positions
            }
            plate_quat = final_positions.get("plate_quat")
        else:
            plate_pos = self.last_plate_pos
            orange_positions = self.last_orange_positions
            plate_quat = self.tracker._plate_quat
        placed_oranges = {
            name
            for name, pos in orange_positions.items()
            if plate_pos is not None and self._is_in_plate_for_scene(plate_pos, pos, plate_quat)
        }
        return self._scene(plate_pos, orange_positions, placed_oranges=placed_oranges, mark_unplaced=True)

    def _scene(self, plate_pos, orange_positions, placed_oranges=None, mark_unplaced=False):
        placed_oranges = set(placed_oranges or ())
        oranges = {}
        for name, pos in sorted(orange_positions.items()):
            oranges[name] = {
                "label": self._label(name),
                "position": self._vec(pos),
                "status": "placed" if name in placed_oranges else "unplaced" if mark_unplaced else "unknown",
            }
        return {
            "plate_position": self._vec(plate_pos) if plate_pos is not None else None,
            "oranges": oranges,
        }

    def _is_in_plate_for_scene(self, plate_pos, orange_pos, plate_quat):
        positions = {"plate": plate_pos, "plate_quat": plate_quat}
        if is_plate_upside_down(positions):
            return False
        _, _, in_plate = plate_position_metrics(
            plate_pos,
            orange_pos,
            plate_radius=self.tracker.PLATE_RADIUS,
            plate_z_min=self.tracker.PLATE_Z_MIN,
            plate_z_max=self.tracker.PLATE_Z_MAX,
            plate_quat=plate_quat,
            inner_radius=self.tracker.PLATE_INNER_RADIUS,
            cone_height=self.tracker.PLATE_CONE_HEIGHT,
        )
        return in_plate

    def _label(self, orange_name):
        if orange_name is None:
            return None
        return self.initial_labels.get(orange_name, orange_name)

    @classmethod
    def _vec(cls, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().flatten().tolist()
        elif isinstance(value, np.ndarray):
            value = value.flatten().tolist()
        return [round(float(v), 5) for v in value]

    def _live(self, episode, step, phase, detail):
        if not self.live_enabled or step % self.live_every_steps != 0:
            return
        sys.stdout.write(f"\r\033[2K[Ep {episode:02d} | Step {step:04d}] {phase}: {detail}")
        sys.stdout.flush()
        self._live_active = True

    def _print_event(self, episode, step, phase, detail):
        if self._live_active:
            sys.stdout.write("\r\033[2K")
            self._live_active = False
        print(f"[Ep {episode:02d} | Step {step:04d}] {phase}: {detail}", flush=True)
