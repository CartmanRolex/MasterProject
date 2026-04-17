import numpy as np
import torch

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_REST_POSE_RANGE, SO101_FOLLOWER_USD_JOINT_LIMLITS

from ..device_base import Device
from .gamepad_utils import GamepadController


class SO101GamepadV3(Device):
    """Gamepad controller for sending SE(3) commands as delta poses for so101 single arm.

    This keeps gamepad_v2 feel and adds roll lock toggling at runtime.
    Wrist roll is controlled directly when unlocked. When lock_roll=True
    (default), right stick X input is ignored. Press X to toggle the lock.

    Key bindings:
    ====================== ==========================================
    Description              Key
    ====================== ==========================================
    Forward / Backward       Left stick Y (forward / backward)
    Shoulder Pan Left/Right  Left stick X (left / right)
    Up / Down                Right stick Y (up / down)
    Pitch Up/Down            LB / LT
    Wrist Roll Left/Right    Right stick X (when unlocked)
    Gripper Open / Close     RT / RB
    Toggle Roll Lock         X button
    ====================== ==========================================
    """

    # Per-axis sensitivity defaults.
    FORWARD_BACKWARD_SENSITIVITY = 0.01
    UP_DOWN_SENSITIVITY = 0.01
    SHOULDER_PAN_SENSITIVITY = 0.03
    PITCH_SENSITIVITY = 0.15
    WRIST_ROLL_SENSITIVITY = 0.15
    GRIPPER_SENSITIVITY = 0.15

    def __init__(
        self,
        env,
        sensitivity: float = 1.0,
        lock_roll: bool = True,
        deadzone: float = 0.08,
        debug_print_inputs: bool = False,
    ):
        self.lock_roll = lock_roll
        self.debug_print_inputs = debug_print_inputs
        super().__init__(env, "gamepad_v3")

        # Store per-axis sensitivity values (scaled by global sensitivity multiplier).
        self.forward_backward_sensitivity = self.FORWARD_BACKWARD_SENSITIVITY * sensitivity
        self.up_down_sensitivity = self.UP_DOWN_SENSITIVITY * sensitivity
        self.shoulder_pan_sensitivity = self.SHOULDER_PAN_SENSITIVITY * sensitivity
        self.pitch_sensitivity = self.PITCH_SENSITIVITY * sensitivity
        self.wrist_roll_sensitivity = self.WRIST_ROLL_SENSITIVITY * sensitivity
        self.gripper_sensitivity = self.GRIPPER_SENSITIVITY * sensitivity

        # initialize gamepad controller
        self._gamepad = GamepadController(deadzone=deadzone)
        self._gamepad.start()
        self._b_override_active = False
        self._last_x_button_state = False
        if "xbox" not in self._gamepad.name:
            raise ValueError("Only Xbox gamepads are supported. Please connect an Xbox gamepad and try again.")
        self._create_key_mapping()
        self._action_update_list = [self._update_arm_action]

        # command buffers (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_wrist_roll, d_gripper)
        self._delta_action = np.zeros(9)

        # initialize the target frame for frame-relative control
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]
        self._joint_names = self.robot_asset.data.joint_names
        self._shoulder_pan_joint_idx = self._joint_names.index("shoulder_pan")
        self._wrist_roll_joint_idx = self._joint_names.index("wrist_roll")
        self._gripper_joint_idx = self._joint_names.index("gripper")
        shoulder_pan_limits_deg = SO101_FOLLOWER_USD_JOINT_LIMLITS["shoulder_pan"]
        wrist_roll_limits_deg = SO101_FOLLOWER_USD_JOINT_LIMLITS["wrist_roll"]
        self._shoulder_pan_min = np.deg2rad(shoulder_pan_limits_deg[0])
        self._shoulder_pan_max = np.deg2rad(shoulder_pan_limits_deg[1])
        self._wrist_roll_min = np.deg2rad(wrist_roll_limits_deg[0])
        self._wrist_roll_max = np.deg2rad(wrist_roll_limits_deg[1])
        self._gripper_min = np.deg2rad(SO101_FOLLOWER_USD_JOINT_LIMLITS["gripper"][0])
        self._gripper_max = np.deg2rad(SO101_FOLLOWER_USD_JOINT_LIMLITS["gripper"][1])
        self._shoulder_pan_target = 0.0
        self._wrist_roll_target = 0.0
        self._rest_pose_rad = {
            name: np.deg2rad((bounds[0] + bounds[1]) * 0.5) for name, bounds in SO101_FOLLOWER_REST_POSE_RANGE.items()
        }
        self._gripper_rest_target = float(np.clip(self._rest_pose_rad["gripper"], self._gripper_min, self._gripper_max))
        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]
        self._sync_internal_targets_from_state()

    def __del__(self):
        """Release the gamepad interface."""
        super().__del__()
        self._gamepad.stop()

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["Left Stick Y (forward)", "forward"])
        self._display_controls_table.add_row(["Left Stick Y (backward)", "backward"])
        self._display_controls_table.add_row(["Left Stick X (left)", "shoulder_pan_left"])
        self._display_controls_table.add_row(["Left Stick X (right)", "shoulder_pan_right"])
        self._display_controls_table.add_row(["Right Stick Y (up)", "up"])
        self._display_controls_table.add_row(["Right Stick Y (down)", "down"])
        self._display_controls_table.add_row(["LB", "pitch_up"])
        self._display_controls_table.add_row(["LT", "pitch_down"])
        self._display_controls_table.add_row(["RT", "gripper_open"])
        self._display_controls_table.add_row(["RB", "gripper_close"])
        self._display_controls_table.add_row(["Gamepad B", "send_rest_pose_targets"])
        self._display_controls_table.add_row(["X", "toggle_roll_lock"])
        if not self.lock_roll:
            self._display_controls_table.add_row(["Right Stick X (left)", "wrist_roll_left"])
            self._display_controls_table.add_row(["Right Stick X (right)", "wrist_roll_right"])

    def get_device_state(self):
        delta_action = self._delta_action.copy()
        # Make Cartesian commands roll-invariant by canceling current wrist roll in
        # the target frame before converting to root frame. This applies to both
        # translation (0:3) and rotation (3:6) command vectors.
        wrist_roll_now = float(self.robot_asset.data.joint_pos[0, self._wrist_roll_joint_idx].item())
        c = float(np.cos(-wrist_roll_now))
        s = float(np.sin(-wrist_roll_now))

        # Rotate translation x/y.
        pos_x, pos_y = float(delta_action[0]), float(delta_action[1])
        delta_action[0] = c * pos_x - s * pos_y
        delta_action[1] = s * pos_x + c * pos_y

        # Rotate orientation command x/y (Euler delta vector in target frame).
        rot_x, rot_y = float(delta_action[3]), float(delta_action[4])
        delta_action[3] = c * rot_x - s * rot_y
        delta_action[4] = s * rot_x + c * rot_y
        return self._convert_delta_from_frame(delta_action)

    def reset(self):
        self._delta_action[:] = 0.0
        self._sync_internal_targets_from_state()

    def advance(self):
        self._delta_action[:] = 0.0
        self._update_action()
        return super().advance()

    def _create_key_mapping(self):
        self._ACTION_DELTA_MAPPING = {
            # IK-controlled axes (indices 0-5): position + pitch
            "forward": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.forward_backward_sensitivity,
            "backward": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.forward_backward_sensitivity,
            "up": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.up_down_sensitivity,
            "down": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.up_down_sensitivity,
            "pitch_up": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.pitch_sensitivity,
            "pitch_down": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * self.pitch_sensitivity,
            # Direct-control axes (indices 6-8): shoulder_pan, wrist_roll, gripper
            "shoulder_pan_left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.shoulder_pan_sensitivity,
            "shoulder_pan_right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.shoulder_pan_sensitivity,
            "wrist_roll_left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.wrist_roll_sensitivity,
            "wrist_roll_right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.wrist_roll_sensitivity,
            "gripper_open": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.gripper_sensitivity,
            "gripper_close": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.gripper_sensitivity,
        }
        self._INPUT_KEY_MAPPING_LIST = [
            ("forward", "L_Y", True),
            ("backward", "L_Y"),
            ("shoulder_pan_left", "L_X", True),
            ("shoulder_pan_right", "L_X"),
            ("up", "R_Y", True),
            ("down", "R_Y"),
            ("pitch_up", "LB"),
            ("pitch_down", "LT"),
            ("gripper_open", "RT"),
            ("gripper_close", "RB"),
        ]
        if not self.lock_roll:
            self._INPUT_KEY_MAPPING_LIST += [
                ("wrist_roll_left", "R_X", True),
                ("wrist_roll_right", "R_X"),
            ]

    def _update_action(self):
        self._gamepad.update()
        for update_func in self._action_update_list:
            update_func()

    def _sync_internal_targets_from_state(self):
        shoulder_pan_now = float(self.robot_asset.data.joint_pos[0, self._shoulder_pan_joint_idx].item())
        wrist_roll_now = float(self.robot_asset.data.joint_pos[0, self._wrist_roll_joint_idx].item())
        self._shoulder_pan_target = float(np.clip(shoulder_pan_now, self._shoulder_pan_min, self._shoulder_pan_max))
        self._wrist_roll_target = float(np.clip(wrist_roll_now, self._wrist_roll_min, self._wrist_roll_max))

    def _sync_action_manager_hold_targets_from_state(self):
        """Re-seed action-term hold caches from current joint state to avoid snap-back."""
        action_manager = getattr(self.env, "action_manager", None)
        if action_manager is None:
            return
        terms = getattr(action_manager, "_terms", None)
        if not isinstance(terms, dict):
            return
        arm_term = terms.get("arm_action")
        if arm_term is None:
            return

        if hasattr(arm_term, "_held_joint_pos_des") and hasattr(arm_term, "_joint_ids") and hasattr(arm_term, "_asset"):
            arm_term._held_joint_pos_des = arm_term._asset.data.joint_pos[:, arm_term._joint_ids].clone()
        if hasattr(arm_term, "_target_ee_quat"):
            # Force a fresh orientation capture on next step.
            arm_term._target_ee_quat = None

    def _send_rest_pose_targets(self):
        """Override with absolute rest-pose targets for all known SO101 joints."""
        current = self.robot_asset.data.joint_pos.clone()
        joint_ids = []
        for joint_name, target_rad in self._rest_pose_rad.items():
            if joint_name not in self._joint_names:
                continue
            joint_idx = self._joint_names.index(joint_name)
            current[:, joint_idx] = target_rad
            joint_ids.append(joint_idx)
        if joint_ids:
            # Hard override joint state to bypass IK/action-manager effects while B is held.
            self.robot_asset.write_joint_state_to_sim(
                position=current,
                velocity=torch.zeros_like(current),
            )
            # Also set matching position targets for consistency with actuator controllers.
            self.robot_asset.set_joint_position_target(current[:, joint_ids], joint_ids=joint_ids)

        # Keep internal direct-control targets synchronized with the requested rest pose.
        self._shoulder_pan_target = float(np.clip(self._rest_pose_rad["shoulder_pan"], self._shoulder_pan_min, self._shoulder_pan_max))
        self._wrist_roll_target = float(np.clip(self._rest_pose_rad["wrist_roll"], self._wrist_roll_min, self._wrist_roll_max))
        gripper_now = float(self.robot_asset.data.joint_pos[0, self._gripper_joint_idx].item())
        self._delta_action[8] = self._gripper_rest_target - gripper_now

    def _print_input_debug(self, controller_state):
        """Print current raw input values for deadzone tuning."""
        button_items = sorted(self._gamepad.mappings["buttons"].items(), key=lambda item: item[1])
        axis_items = sorted(self._gamepad.mappings["axes"].items(), key=lambda item: item[1])

        button_state_str = " ".join(
            f"{name}:{int(controller_state.buttons[idx])}" for name, idx in button_items
        )
        axis_state_str = " ".join(
            f"{name}:{controller_state.axes[idx]:+.3f}" for name, idx in axis_items
        )
        print(f"[gamepad_v3] buttons[{button_state_str}] axes[{axis_state_str}]")

    def _update_arm_action(self):
        """Update the delta action based on the gamepad state."""
        controller_state = self._gamepad.get_state()
        if self.debug_print_inputs:
            self._print_input_debug(controller_state)

        # Toggle roll lock on X button press (edge-triggered).
        buttons_mapping = self._gamepad.mappings["buttons"]
        b_pressed = "B" in buttons_mapping and bool(controller_state.buttons[buttons_mapping["B"]])
        if b_pressed:
            self._send_rest_pose_targets()
            self._b_override_active = True
            # Skip normal teleop command generation for this frame while override is active.
            self._delta_action[:] = 0.0
            return
        if self._b_override_active:
            # B was just released: freeze holds at current pose instead of old pre-B target.
            self._sync_internal_targets_from_state()
            self._sync_action_manager_hold_targets_from_state()
            self._b_override_active = False
            self._delta_action[:] = 0.0
            return

        if "X" in buttons_mapping:
            x_pressed = bool(controller_state.buttons[buttons_mapping["X"]])
            if x_pressed and not self._last_x_button_state:
                self.lock_roll = not self.lock_roll
                self._create_key_mapping()
            self._last_x_button_state = x_pressed

        axes_mapping = self._gamepad.mappings["axes"]
        left_stick_filtered = None
        if "L_X" in axes_mapping and "L_Y" in axes_mapping:
            l_x_raw = controller_state.axes[axes_mapping["L_X"]]
            l_y_raw = controller_state.axes[axes_mapping["L_Y"]]
            max_abs = max(abs(l_x_raw), abs(l_y_raw))

            # Keep only the dominant left-stick axis to suppress cross-axis bleed.
            if abs(l_x_raw) >= abs(l_y_raw):
                left_stick_filtered = {"L_X": np.sign(l_x_raw) * max_abs, "L_Y": 0.0}
            else:
                left_stick_filtered = {"L_X": 0.0, "L_Y": np.sign(l_y_raw) * max_abs}

        for input_key_mapping in self._INPUT_KEY_MAPPING_LIST:
            action_name, controller_name = input_key_mapping[0], input_key_mapping[1]
            reverse = input_key_mapping[2] if len(input_key_mapping) > 2 else False
            if controller_name in axes_mapping:
                if left_stick_filtered is not None and controller_name in left_stick_filtered:
                    axis_value = left_stick_filtered[controller_name]
                else:
                    axis_value = controller_state.axes[axes_mapping[controller_name]]
                # Split one signed axis into two one-sided strengths using reverse.
                strength = max(-axis_value, 0.0) if reverse else max(axis_value, 0.0)
                if strength > 0.0:
                    self._delta_action += self._ACTION_DELTA_MAPPING[action_name] * strength
            else:
                is_activate, is_pressed = self._gamepad.lookup_controller_state(controller_state, controller_name, reverse)
                if is_activate and is_pressed:
                    self._delta_action += self._ACTION_DELTA_MAPPING[action_name]

        # Shoulder pan and wrist roll use internal bounded targets to avoid drift from open-loop deltas.
        shoulder_pan_now = float(self.robot_asset.data.joint_pos[0, self._shoulder_pan_joint_idx].item())
        wrist_roll_now = float(self.robot_asset.data.joint_pos[0, self._wrist_roll_joint_idx].item())
        self._shoulder_pan_target += float(self._delta_action[6])
        self._wrist_roll_target += float(self._delta_action[7])
        self._shoulder_pan_target = float(
            np.clip(self._shoulder_pan_target, self._shoulder_pan_min, self._shoulder_pan_max)
        )
        self._wrist_roll_target = float(
            np.clip(self._wrist_roll_target, self._wrist_roll_min, self._wrist_roll_max)
        )
        # Convert absolute targets back to the relative-joint action expected downstream.
        self._delta_action[6] = self._shoulder_pan_target - shoulder_pan_now
        self._delta_action[7] = self._wrist_roll_target - wrist_roll_now
