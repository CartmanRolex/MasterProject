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

    def test_grasp_lift_place_success_sequence(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.assertEqual(self.monitor.subtask_attempts[0]["result"], "success")
        self.monitor.tracker._gripper_tip = vec(0.02, 0.0, 0.07)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.07)
        self.monitor._update_lift(0, 2, 0.0, {"Orange001": vec(0.02, 0.0, 0.07)})
        self.monitor.tracker._gripper_tip = vec(0.0, 0.0, 0.08)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.08)
        self.monitor._update_place(0, 3, vec(0.0, 0.0, 0.0), {"Orange001": vec(0.02, 0.0, 0.03)}, 1.0)
        self.assertEqual([a["subtask"] for a in self.monitor.subtask_attempts], ["GRASP", "LIFT", "PLACE"])
        self.assertTrue(all(a["result"] == "success" for a in self.monitor.subtask_attempts))

    def test_dropped_during_lift(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.monitor.tracker._gripper_tip = vec(0.5, 0.0, 0.0)
        self.monitor.tracker._jaw_tip = vec(0.54, 0.0, 0.0)
        self.monitor._update_lift(0, 2, 0.0, {"Orange001": vec(0.02, 0.0, 0.02)})
        self.assertEqual(self.monitor.subtask_attempts[-1]["failure_reason"], "dropped_during_lift")

    def test_place_can_succeed_without_lift_confirmation(self):
        self.monitor._update_grasp(0, 1, *self.grasp_inputs())
        self.monitor.tracker._gripper_tip = vec(0.0, 0.0, 0.08)
        self.monitor.tracker._jaw_tip = vec(0.04, 0.0, 0.08)
        self.monitor._update_lift(0, 2, 1.0, {"Orange001": vec(0.02, 0.0, 0.03)})
        self.assertEqual([a["subtask"] for a in self.monitor.subtask_attempts], ["GRASP", "LIFT", "PLACE"])
        self.assertEqual(self.monitor.subtask_attempts[1]["result"], "skipped")
        self.assertEqual(self.monitor.subtask_attempts[1]["failure_reason"], "inferred_place_without_lift")
        self.assertEqual(self.monitor.subtask_attempts[2]["result"], "success")
        self.assertTrue(self.monitor.subtask_attempts[2]["metrics"]["inferred_place_without_lift"])

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
