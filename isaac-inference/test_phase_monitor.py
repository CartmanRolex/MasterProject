import os
import unittest

import torch

from phase_monitor import PhaseMonitor


def vec(x, y, z):
    return torch.tensor([x, y, z], dtype=torch.float32)


class PhaseMonitorTraceTest(unittest.TestCase):
    def setUp(self):
        os.environ["PHASE_DEBUG_LIVE"] = "0"
        self.monitor = PhaseMonitor(model_id="test-model")
        self.monitor.tracker.patience_frames = 1
        self.monitor.tracker.stability_frames = 1
        self.monitor.retarget_frames = 2
        self.monitor.lift_start_frames = 1
        self.monitor.initial_labels = {
            "Orange001": "left",
            "Orange002": "right",
            "Orange003": "middle",
        }
        self.monitor.initial_orange_z = {
            "Orange001": 0.0,
            "Orange002": 0.0,
            "Orange003": 0.0,
        }
        self.monitor.initial_scene = self.monitor._scene(
            vec(0.0, 0.0, 0.0),
            {
                "Orange001": vec(0.02, 0.0, 0.0),
                "Orange002": vec(0.20, 0.0, 0.0),
                "Orange003": vec(0.40, 0.0, 0.0),
            },
        )
        self.monitor.last_plate_pos = vec(0.0, 0.0, 0.0)
        self.monitor.last_orange_positions = {
            "Orange001": vec(0.02, 0.0, 0.0),
            "Orange002": vec(0.20, 0.0, 0.0),
            "Orange003": vec(0.40, 0.0, 0.0),
        }
        self.monitor.tracker._plate_quat = None
        self.monitor.tracker._gripper_tip = vec(0.0, 0.0, 0.0)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.0)

    def grasp_inputs(self, orange_positions=None, ready=True):
        force = 20.0 if ready else 0.0
        return (
            vec(0.0, 0.0, 0.0),
            vec(0.04, 0.0, 0.0),
            orange_positions
            or {
                "Orange001": vec(0.02, 0.0, 0.0),
                "Orange002": vec(0.20, 0.0, 0.0),
                "Orange003": vec(0.40, 0.0, 0.0),
            },
            vec(force, 0.0, 0.0),
            vec(force, 0.0, 0.0),
        )

    def test_stable_inferred_grasp_target(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs(ready=False))
        self.assertEqual(self.monitor._active_attempt["subtask"], "GRASP")
        self.assertEqual(self.monitor._active_attempt["inferred_target_orange"], "Orange001")
        self.assertEqual(self.monitor._active_attempt["requested_orange"], None)

    def test_stable_retarget_after_closest_change(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs(ready=False))
        positions = {
            "Orange001": vec(0.30, 0.0, 0.0),
            "Orange002": vec(0.02, 0.0, 0.0),
            "Orange003": vec(0.40, 0.0, 0.0),
        }
        self.monitor._update_grasp(0, 2, *self.grasp_inputs(positions, ready=False))
        self.monitor._update_grasp(0, 3, *self.grasp_inputs(positions, ready=False))
        self.assertEqual(self.monitor.subtask_attempts[0]["result"], "retargeted")
        self.assertEqual(self.monitor.subtask_attempts[0]["failure_reason"], "inferred_target_changed")
        self.assertEqual(self.monitor._active_attempt["inferred_target_orange"], "Orange002")

    def test_inferred_grasp_from_height_when_force_grasp_never_confirms(self):
        # Contact forces never reach the grasp threshold (ready=False), so the force-based
        # grasp confirmation never fires. A held orange (tip proximity) rising above the
        # height threshold should still infer a grasp and start the lift.
        self.monitor.tracker._gripper_tip = vec(0.02, 0.0, 0.03)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.03)
        positions = {
            "Orange001": vec(0.02, 0.0, 0.03),
            "Orange002": vec(0.20, 0.0, 0.0),
            "Orange003": vec(0.40, 0.0, 0.0),
        }
        self.monitor._update_grasp(0, 1, *self.grasp_inputs(positions, ready=False))
        self.assertEqual(self.monitor.current_phase, "LIFTING")
        self.assertTrue(self.monitor.lift_started)
        self.assertEqual([a["subtask"] for a in self.monitor.subtask_attempts], ["GRASP"])
        self.assertEqual(self.monitor.subtask_attempts[0]["result"], "success")
        self.assertTrue(self.monitor.subtask_attempts[0]["metrics"]["inferred_from_lift"])
        self.assertEqual(len([e for e in self.monitor.events if e["event_type"] == "grasp_success"]), 1)
        self.assertEqual(len([e for e in self.monitor.events if e["event_type"] == "lift_started"]), 1)

    def test_held_orange_not_high_enough_stays_searching(self):
        # Held but below the height threshold: no grasp should be inferred.
        self.monitor.tracker._gripper_tip = vec(0.02, 0.0, 0.005)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.005)
        positions = {
            "Orange001": vec(0.02, 0.0, 0.005),
            "Orange002": vec(0.20, 0.0, 0.0),
            "Orange003": vec(0.40, 0.0, 0.0),
        }
        self.monitor._update_grasp(0, 1, *self.grasp_inputs(positions, ready=False))
        self.assertEqual(self.monitor.current_phase, "SEARCHING")
        self.assertFalse(self.monitor.lift_started)

    def test_lift_detected_with_gripper_open_when_held(self):
        # Lift detection must not require gripper closure: a grasped orange props the jaws
        # open. With the orange held and risen (away from the plate), a gripper that reads
        # "open" still lifts.
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.monitor.tracker._gripper_tip = vec(0.30, 0.0, 0.07)
        self.monitor.tracker._jaw_tip = vec(0.34, 0.0, 0.07)
        # gripper_pos=1.0 -> reads "open"; orange held and 0.07 m up (above 0.06 confirm)
        self.monitor._update_lift(0, 2, 1.0, {"Orange001": vec(0.30, 0.0, 0.07)})
        self.assertTrue(self.monitor.lift_started)
        self.assertIn("lift_started", [e["event_type"] for e in self.monitor.events])
        self.assertIn("lift_success", [e["event_type"] for e in self.monitor.events])

    def test_grasp_lift_place_success_sequence(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.assertEqual(self.monitor.subtask_attempts[0]["result"], "success")
        self.monitor.tracker._gripper_tip = vec(0.02, 0.0, 0.07)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.07)
        self.monitor._update_lift(0, 2, 0.0, {"Orange001": vec(0.02, 0.0, 0.07)})
        # Gripper retracts away from the orange (orange no longer held) and the orange rests
        # in the plate -> place confirmed (no requirement that the tip rise to a set height).
        self.monitor.tracker._gripper_tip = vec(0.0, 0.0, 0.25)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.25)
        self.monitor._update_place(0, 3, vec(0.0, 0.0, 0.0), {"Orange001": vec(0.02, 0.0, 0.03)}, 1.0)
        self.assertEqual([a["subtask"] for a in self.monitor.subtask_attempts], ["GRASP", "LIFT", "PLACE"])
        self.assertTrue(all(a["result"] == "success" for a in self.monitor.subtask_attempts))

    def test_dropped_after_grasp_before_lift_starts(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.monitor.tracker._gripper_tip = vec(0.5, 0.0, 0.0)
        self.monitor.tracker._jaw_tip = vec(0.54, 0.0, 0.0)
        self.monitor._update_lift(0, 2, 0.0, {"Orange001": vec(0.02, 0.0, 0.0)})
        self.assertFalse(self.monitor.lift_started)
        self.assertEqual([a["subtask"] for a in self.monitor.subtask_attempts], ["GRASP"])
        failure_events = [e for e in self.monitor.events if e["event_type"] == "grasp_failure"]
        self.assertEqual(len(failure_events), 1)
        self.assertEqual(failure_events[0]["reason"], "dropped_after_grasp")

    def test_dropped_during_lift(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.monitor.tracker._gripper_tip = vec(0.02, 0.0, 0.03)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.03)
        self.monitor._update_lift(0, 2, 0.0, {"Orange001": vec(0.02, 0.0, 0.03)})
        self.assertTrue(self.monitor.lift_started)
        self.monitor.tracker._gripper_tip = vec(0.5, 0.0, 0.0)
        self.monitor.tracker._jaw_tip = vec(0.54, 0.0, 0.0)
        # Orange released far from the plate (not in plate) -> a real dropped lift.
        self.monitor._update_lift(0, 3, 0.0, {"Orange001": vec(0.40, 0.0, 0.02)})
        self.assertEqual(self.monitor.subtask_attempts[-1]["failure_reason"], "dropped_during_lift")

    def test_release_in_plate_during_lift_is_lift_then_place(self):
        # Edge case: while still in LIFTING, the orange leaves the gripper but is resting in
        # the plate. This must NOT be counted as a dropped lift — it is a completed lift
        # followed by a place. (Lift and place happen almost together.)
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.assertEqual(self.monitor.current_phase, "LIFTING")
        self.assertFalse(self.monitor.lift_started)
        # Gripper retracted away from the orange (not held); orange resting in the plate.
        self.monitor.tracker._gripper_tip = vec(0.0, 0.0, 0.25)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.25)
        self.monitor._update_lift(0, 2, 1.0, {"Orange001": vec(0.02, 0.0, 0.03)})
        self.assertNotIn(
            "dropped_during_lift",
            [a.get("failure_reason") for a in self.monitor.subtask_attempts],
        )
        self.assertEqual(self.monitor.current_phase, "PLACING")
        self.assertEqual([a["subtask"] for a in self.monitor.subtask_attempts], ["GRASP", "LIFT"])
        self.assertEqual(self.monitor.subtask_attempts[1]["result"], "success")  # LIFT success
        self.assertIn("lift_success", [e["event_type"] for e in self.monitor.events])
        # The handed-off PLACE attempt now confirms (in plate + not held + stable).
        self.monitor._update_place(0, 3, vec(0.0, 0.0, 0.0), {"Orange001": vec(0.02, 0.0, 0.03)}, 1.0)
        self.assertEqual([a["subtask"] for a in self.monitor.subtask_attempts], ["GRASP", "LIFT", "PLACE"])
        self.assertEqual(self.monitor.subtask_attempts[2]["result"], "success")

    def test_dropped_during_place(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.monitor.tracker._gripper_tip = vec(0.02, 0.0, 0.07)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.07)
        self.monitor._update_lift(0, 2, 0.0, {"Orange001": vec(0.02, 0.0, 0.07)})
        self.monitor.tracker._gripper_tip = vec(0.5, 0.0, 0.08)
        self.monitor.tracker._jaw_tip = vec(0.54, 0.0, 0.08)
        self.monitor._update_place(0, 3, vec(0.0, 0.0, 0.0), {"Orange001": vec(0.30, 0.0, 0.03)}, 1.0)
        self.assertEqual(self.monitor.subtask_attempts[-1]["failure_reason"], "dropped_during_place")

    def test_placed_orange_left_plate_event(self):
        self.monitor.placed_oranges.add("Orange001")
        self.monitor.bounce_frames = 1
        self.monitor.tracker._gripper_tip = vec(0.5, 0.0, 0.08)
        self.monitor.tracker._jaw_tip = vec(0.54, 0.0, 0.08)
        self.monitor._check_placed_bounce(0, 4, vec(0.0, 0.0, 0.0), {"Orange001": vec(0.30, 0.0, 0.03)})
        self.assertEqual(self.monitor.timeline[-1]["reason"], "placed_orange_left_plate")

    def test_episode_end_during_inferred_grasp_search(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs(ready=False))
        record = self.monitor.build_record(
            0,
            10,
            0,
            end_reason="env_truncated",
            is_success=False,
            final_positions={
                "plate": vec(0.0, 0.0, 0.0),
                "plate_quat": None,
                "Orange001": vec(0.20, 0.0, 0.03),
                "Orange002": vec(0.30, 0.0, 0.03),
                "Orange003": vec(0.40, 0.0, 0.03),
            },
        )
        self.assertEqual(record["subtask_attempts"][-1]["failure_reason"], "no_confirmed_progress")
        self.assertEqual(record["phase_debug"]["trace_source"], "flat_observed_physics")


if __name__ == "__main__":
    unittest.main()
