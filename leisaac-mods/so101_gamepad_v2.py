import numpy as np

from ..device_base import Device
from .gamepad_utils import GamepadController


class SO101GamepadV2(Device):
    """Gamepad controller for sending SE(3) commands as delta poses for so101 single arm.

    Key bindings:
    ====================== ==========================================
    Description              Key
    ====================== ==========================================
    Forward / Backward       Left stick Y (forward / backward)
    Left / Right             Left stick X (left / right)
    Up / Down                Right stick Y (forward / backward)
    Rotate (Roll) Left/Right Right stick X (left / right)
    Rotate (Pitch) Up/Down   LB / LT
    Gripper Open / Close     RT / RB
    ====================== ==========================================
    """

    # Per-axis sensitivity defaults.
    FORWARD_BACKWARD_SENSITIVITY = 0.01
    UP_DOWN_SENSITIVITY = 0.01
    LEFT_RIGHT_SENSITIVITY = 0.1
    ROTATE_UP_DOWN_SENSITIVITY = 0.15
    ROTATE_LEFT_RIGHT_SENSITIVITY = 0.15
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
        super().__init__(env, "gamepad_v2")

        # Store per-axis sensitivity values (scaled by global sensitivity multiplier).
        self.forward_backward_sensitivity = self.FORWARD_BACKWARD_SENSITIVITY * sensitivity
        self.up_down_sensitivity = self.UP_DOWN_SENSITIVITY * sensitivity
        self.left_right_sensitivity = self.LEFT_RIGHT_SENSITIVITY * sensitivity
        self.rotate_up_down_sensitivity = self.ROTATE_UP_DOWN_SENSITIVITY * sensitivity
        self.rotate_left_right_sensitivity = self.ROTATE_LEFT_RIGHT_SENSITIVITY * sensitivity
        self.gripper_sensitivity = self.GRIPPER_SENSITIVITY * sensitivity

        # initialize gamepad controller
        self._gamepad = GamepadController(deadzone=deadzone)
        self._gamepad.start()
        self._last_x_button_state = False
        if "xbox" not in self._gamepad.name:
            raise ValueError("Only Xbox gamepads are supported. Please connect an Xbox gamepad and try again.")
        self._create_key_mapping()
        self._action_update_list = [self._update_arm_action]

        # command buffers (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)

        # initialize the target frame for frame-relative control
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]
        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def __del__(self):
        """Release the gamepad interface."""
        super().__del__()
        self._gamepad.stop()

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["Left Stick Y (forward)", "forward"])
        self._display_controls_table.add_row(["Left Stick Y (backward)", "backward"])
        self._display_controls_table.add_row(["Left Stick X (left)", "left"])
        self._display_controls_table.add_row(["Left Stick X (right)", "right"])
        self._display_controls_table.add_row(["Right Stick Y (up)", "up"])
        self._display_controls_table.add_row(["Right Stick Y (down)", "down"])
        self._display_controls_table.add_row(["LB", "rotate_up"])
        self._display_controls_table.add_row(["LT", "rotate_down"])
        self._display_controls_table.add_row(["RT", "gripper_open"])
        self._display_controls_table.add_row(["RB", "gripper_close"])
        if not self.lock_roll:
            self._display_controls_table.add_row(["Right Stick X (left)", "rotate_left"])
            self._display_controls_table.add_row(["Right Stick X (right)", "rotate_right"])

    def get_device_state(self):
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        self._delta_action[:] = 0.0

    def advance(self):
        self._delta_action[:] = 0.0
        self._update_action()
        return super().advance()

    def _create_key_mapping(self):
        self._ACTION_DELTA_MAPPING = {
            "forward": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.forward_backward_sensitivity,
            "backward": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.forward_backward_sensitivity,
            "left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.left_right_sensitivity,
            "right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.left_right_sensitivity,
            "up": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.up_down_sensitivity,
            "down": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.up_down_sensitivity,
            "rotate_up": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.rotate_up_down_sensitivity,
            "rotate_down": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.rotate_up_down_sensitivity,
            "rotate_left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.rotate_left_right_sensitivity,
            "rotate_right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.rotate_left_right_sensitivity,
            "gripper_open": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.gripper_sensitivity,
            "gripper_close": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.gripper_sensitivity,
        }
        self._INPUT_KEY_MAPPING_LIST = [
            ("forward", "L_Y", True),
            ("backward", "L_Y"),
            ("left", "L_X", True),
            ("right", "L_X"),
            ("up", "R_Y", True),
            ("down", "R_Y"),
            ("rotate_up", "LB"),
            ("rotate_down", "LT"),
            ("gripper_open", "RT"),
            ("gripper_close", "RB"),
        ]
        if not self.lock_roll:
            self._INPUT_KEY_MAPPING_LIST += [
                ("rotate_left", "R_X", True),
                ("rotate_right", "R_X"),
            ]

    def _update_action(self):
        self._gamepad.update()
        for update_func in self._action_update_list:
            update_func()

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
        print(f"[gamepad_v2] buttons[{button_state_str}] axes[{axis_state_str}]")

    def _update_arm_action(self):
        """Update the delta action based on the gamepad state."""
        controller_state = self._gamepad.get_state()
        if self.debug_print_inputs:
            self._print_input_debug(controller_state)

        # Toggle roll lock on X button press (edge-triggered)
        buttons_mapping = self._gamepad.mappings["buttons"]
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
