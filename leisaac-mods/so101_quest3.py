"""Meta Quest 3 hand-tracking teleoperation device for the SO-101 arm.

This reuses the WebXR streaming layer authored by Gregorio for the stock Isaac
Lab Franka demo (``isaac-inference/gregorio/quest3_device.py``): an aiohttp
server serves a WebXR page to the Quest's Meta Browser, the page streams the
right-hand joint poses + wrist quaternion over a WebSocket, and a thumb-index
pinch drives the gripper.

The Isaac-free, calibration-critical pieces (the WebXR page ``_HTML_TEMPLATE``,
the ``_R_XR_TO_ISAAC`` frame matrix, ``_TeleopState``, the pinch indices, and the
``xr_delta_to_world`` delta math) live in ``quest3_webxr.py`` so they stay
identical between this device and the standalone calibration tool
(``leisaac-mods/quest3_hand_monitor.py``). Anything tuned with the monitor
therefore transfers here unchanged.

The action math is rewritten for leisaac: instead of Gregorio's *absolute* 8D
IK target ``[pos, quat, gripper]`` for a 7-DOF Franka, this emits a *per-step
delta* in the SO-101 robot **base/root frame** that flows through leisaac's
relative DLS-IK action term (see ``init_action_cfg``/``preprocess_device_action``
for the matching ``"quest3"`` branches):

    joint_state = [dx, 0, dz, drvx, drvy, drvz, d_shoulder_pan, d_gripper]  # 8D, root frame

Like ``so101_gamepad_v3``, the **lateral** (left/right) hand motion is decomposed
off the IK and drives the ``shoulder_pan`` joint directly (a bounded relative
target), while forward/back + up/down + the wrist rotation go to a DLS IK
controller over the remaining 4 arm joints (shoulder_lift, elbow_flex,
wrist_flex, wrist_roll) — ``shoulder_pan`` is removed from the IK. So the root-Y
component of the EE position delta is zeroed (``dy = 0``) and re-routed to
``d_shoulder_pan``. ``d_gripper`` is a relative joint command (pinch closes).

Start/reset use leisaac's standard desktop keyboard: B starts control, R resets
(fail), N resets (success). The in-headset blue button only starts the WebXR
hand-tracking session.

Tuning: ``pos_scale``, ``gripper_sensitivity``, ``pinch_threshold_m``, and the
per-step clamps. The hand->arm axis directions are set by ``_R_XR_TO_ISAAC_SO101``
(defined in this module and passed to ``xr_delta_to_world`` as ``R``): it maps the
XR wrist delta into the Isaac **world** frame so that ``_world_to_root`` can hand the
relative DLS-IK term a true root-frame delta, and the robot moves up / right / back
exactly as the calibration monitor and ``so101_gamepad_v3`` name those axes. If a
motion axis comes out mirrored or swapped on the robot, flip the corresponding row's
sign of that matrix. It is the device's own copy, so the calibration monitor (which
uses ``quest3_webxr._R_XR_TO_ISAAC``) is unaffected.

Rotation is **absolute and anchored** (not a per-step delta): each step the device
feeds the relative-IK term the root-frame error rotvec from the current gripper
orientation to a target = the gripper's anchor orientation composed with the hand's
rotation-since-anchor (mapped via ``_R_rot_map``). The IK's relative mode applies
that as a world/root-frame left-multiplication, so the gripper chases the hand's
absolute orientation every step — drift-free, no delta accumulation, and no
sign/swap hacks. The anchor (hand + gripper orientation) is captured on start /
reset / tracking-loss and held until the next reset, so the gripper mirrors the
hand's rotation from its start pose with no start jump (any constant hand<->gripper
orientation offset auto-cancels at the anchor). ``rot_scale`` does not apply to
rotation (the error is the real orientation error, 1:1); ``max_rot_step_rad`` is the
rate cap. If a rotation axis comes out swapped/wrong, ``_R_rot_map`` is the single
tuning knob — it defaults to the same matrix as the position path but may be split
into a dedicated one.

Axis fixes happen in ``_anchored_rotation_error``: the hand's forward<->right
rotation is **swapped on the commanded target** (feedforward, so the closed loop
stays stable — permuting the error instead diverges). Three further corrections,
**all verified in a numerical simulation of the function**, remove the cross-axis
bleed seen on hardware — one on the hand side, two on the gripper side:

  1. **Hand heading stripped (swing-twist), hand side.** The command is the
     axis-angle of the hand's rotation-since-anchor, and the axis-angle of a
     compound rotation does not separate cleanly: a 90° hand *yaw* rotates a forward
     *roll* into the right axis (seen in the calibration monitor as the "right"
     value moving while the hand only rolls). A gimbal-free swing-twist split about
     vertical removes the heading twist (owned by shoulder_pan), leaving pitch+roll.
  2. The chase **target is re-headed to the gripper's current heading** (rotated by
     the pan-since-anchor about root up). Left-multiplying the root-frame command
     onto the anchor leaves a large heading error in the chase rotvec once the arm
     pans, which bleeds a forward hand rotation into the gripper's right axis.
  3. The **gripper's own up axis is freed, not root up**: heading is owned by
     shoulder_pan, so we project the gripper-up component (body +x) out of the
     error. When the wrist is pitched, gripper-up tilts away from root up, so the
     old "zero root-Z" dropped the wrong component and bled a forward roll into
     pitch.

In simulation a single-axis hand rotation (roll/pitch) now produces a single
constant-axis gripper rotation at every hand yaw and every gripper pan/pitch, and a
hand yaw produces none. The naive versions only looked correct at the start, where
hand and gripper are unyawed/unpanned/unpitched so all the frames coincide.
"""

import asyncio
import json
import threading

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from aiohttp import WSMsgType, web
from scipy.spatial.transform import Rotation

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

from .device_base import Device
from .quest3_webxr import (
    _HTML_TEMPLATE,
    _R_XR_TO_ISAAC,  # noqa: F401  (re-exported for callers/tools that import it from here)
    _TeleopState,
    pinch_distance,
)


# Per-step XR->Isaac-WORLD axis map used by the *device* (passed to
# ``xr_delta_to_world`` as ``R``). ``get_device_state`` runs the result through
# ``_world_to_root`` (which rotates by the live robot base quaternion), so this
# matrix MUST land the XR wrist delta in the **Isaac world** frame (Z-up, +X fwd)
# — NOT the monitor's [up, right, back] display frame. It is the world-frame
# companion of the monitor's verified ``quest3_webxr._R_XR_TO_ISAAC``: the monitor
# maps XR wrist axes [right, up, back] -> the gamepad_v3 display axes
# [up, right, back]; relabelling those into Isaac world (forward = -back,
# left = -right, up = up) yields this matrix. So with XR wrist axes [right, up, back]:
#     world +X (forward) <- -wrist[2]   (hand back  -> -X)
#     world +Y (left)    <- -wrist[0]   (hand right -> -Y)
#     world +Z (up)      <-  wrist[1]   (hand up    -> +Z)
# The robot then moves up / right / back exactly as the monitor and
# so101_gamepad_v3 name those axes. Kept as the device's own copy so the
# calibration monitor (its own ``_R_XR_TO_ISAAC``) is unaffected. This matrix is
# ALSO reused as the rotation map (``_R_rot_map = Rotation.from_matrix(...)``),
# which requires a proper rotation (det = +1) — so do NOT fix a mirrored
# translation axis by flipping a single row here (that makes det = -1 and
# scipy rejects it). Position-only sign fixes go on the root-frame output in
# ``get_device_state`` instead (see the forward/back negation on ``out[0]``).
_R_XR_TO_ISAAC_SO101: np.ndarray = np.array(
    [[0.0, 0.0, -1.0],
     [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


class SO101Quest3(Device):
    """Quest 3 hand tracking -> SO-101 relative IK teleoperation.

    ``get_device_state`` returns an 8D root-frame delta:
    ``[dx, 0, dz, drvx, drvy, drvz, d_shoulder_pan, d_gripper]`` — lateral motion
    drives ``shoulder_pan`` directly, the rest goes to the IK.
    """

    def __init__(
        self,
        env,
        sensitivity: float = 1.0,
        port: int = 8080,
        send_hz: int = 60,
        pinch_threshold_m: float = 0.035,
        safety_timeout_s: float = 0.5,
        max_pos_step_m: float = 0.02,
        max_rot_step_rad: float = 0.10,
        rot_track_gain: float = 0.4,
        fwd_up_gain: float = 1.0,
        gripper_sensitivity: float = 0.15,
        shoulder_pan_sensitivity: float = 4.0,
    ) -> None:
        # Per-step scales. sensitivity is a global multiplier (matches gamepad devices).
        self.pos_scale = 1.0 * sensitivity
        self.rot_scale = 1.0 * sensitivity
        self.gripper_sensitivity = gripper_sensitivity * sensitivity
        # rad of shoulder_pan per metre of lateral (root-Y) hand motion.
        self.shoulder_pan_sensitivity = shoulder_pan_sensitivity * sensitivity
        self.pinch_threshold_m = pinch_threshold_m
        self.safety_timeout_s = safety_timeout_s
        self.max_pos_step_m = max_pos_step_m
        self.max_rot_step_rad = max_rot_step_rad
        # Extra multiplier on the IK-bound translation axes ONLY (forward/back, up/down
        # — gripper-frame fwd_cmd/up_cmd), applied after pos_scale/clamp. Left/right
        # (shoulder_pan) is untouched, tuned separately by shoulder_pan_sensitivity.
        # Raise this to make the arm travel faster forward/back and up/down per metre
        # of hand motion, without changing pan speed or hand-motion->command clamping.
        self.fwd_up_gain = fwd_up_gain
        # Softness of orientation tracking, in (0, 1]. The DLS IK weights position
        # and rotation error equally and the same arm joints serve both, so on this
        # 4-joint arm a hard orientation command starves translation. Scaling the
        # rotation error down de-weights it: lower = orientation gives way to
        # translation (it still converges once you stop translating); 1.0 = track
        # orientation as hard as possible (original behaviour, translation suffers).
        self.rot_track_gain = rot_track_gain

        super().__init__(env, "quest3")

        # robot handle (root frame for delta conversion)
        self.robot_asset = self.env.scene["robot"]

        # shoulder_pan is driven directly (not by the IK): bounded internal target
        # accumulated from lateral hand motion, like so101_gamepad_v3.
        self._joint_names = self.robot_asset.data.joint_names
        self._shoulder_pan_joint_idx = self._joint_names.index("shoulder_pan")
        shoulder_pan_limits_deg = SO101_FOLLOWER_USD_JOINT_LIMLITS["shoulder_pan"]
        self._shoulder_pan_min = float(np.deg2rad(shoulder_pan_limits_deg[0]))
        self._shoulder_pan_max = float(np.deg2rad(shoulder_pan_limits_deg[1]))
        self._shoulder_pan_target = self._read_shoulder_pan()

        # body the IK targets (``body_name="gripper"``) — read each step for the
        # current gripper orientation used by the absolute-rotation error term.
        self._gripper_body_idx = int(self.robot_asset.find_bodies("gripper")[0][0])

        # XR-wrist-rotation -> world-rotation axis map as a scipy Rotation (det = +1,
        # a proper rotation). Used to re-express the hand's rotation-since-anchor
        # into the world frame (conjugation = the quaternion form of R @ rotvec) for
        # the absolute-rotation target. Defaults to the same matrix as the position
        # path; if a rotation axis comes out swapped/wrong on hardware, tune this
        # (or split it into a dedicated matrix) — the anchored offset absorbs any
        # *constant* offset, so only the axis assignment matters here.
        self._R_rot_map = Rotation.from_matrix(_R_XR_TO_ISAAC_SO101)

        # previous wrist pose, captured on first valid frame (and after every reset)
        self._prev_wrist_pos: np.ndarray | None = None
        self._prev_wrist_rot: Rotation | None = None
        # Anchor wrist orientation for orientation-aware translation. Unlike
        # _prev_wrist_rot (re-anchored every step for the rotation delta), this is
        # held until an explicit reset (and on tracking loss) so the translation
        # frame follows the hand's pointing direction across a whole session
        # ("additive"): the room-frame position delta is re-expressed in the hand
        # body frame via dQ = rot_now * anchor_wrist_rot.inv() before mapping.
        self._anchor_wrist_rot: Rotation | None = None
        # Anchor gripper (EE) orientation in the Isaac world frame, captured at the
        # same moment as _anchor_wrist_rot. The absolute-rotation IK target is
        # Q_anchor_ee_world composed with the hand's rotation-since-anchor, so the
        # gripper mirrors the hand's rotation from its start pose (no start jump,
        # constant hand<->gripper orientation offsets auto-cancel at the anchor).
        self._anchor_ee_world_rot: Rotation | None = None
        self._close_gripper = False

        # WebXR streaming server (reused from gregorio via quest3_webxr)
        self._cfg_port = port
        self._state = _TeleopState()
        self._html = _HTML_TEMPLATE.replace("__SEND_INTERVAL_MS__", str(1000 // send_hz))
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True, name="quest3-ws"
        )
        self._server_thread.start()
        print(f"[Quest3] Open the ngrok https URL in the Meta Browser, then press the blue button.")

    def __str__(self) -> str:
        return (
            f"SO101Quest3  port={self._cfg_port}\n"
            f"\tpos_scale={self.pos_scale}  rot_scale={self.rot_scale}\n"
            f"\tpinch_threshold={self.pinch_threshold_m:.3f} m  (relative IK)"
        )

    def reset(self) -> None:
        # re-anchor on next valid frame so teleop resumes from the current pose
        self._prev_wrist_pos = None
        self._prev_wrist_rot = None
        # also drop the translation-frame + rotation anchors so the hand's current
        # pose redefines "forward/right/up" (position) and the gripper's current
        # orientation becomes the rotation zero point, after a reset
        self._anchor_wrist_rot = None
        self._anchor_ee_world_rot = None
        self._close_gripper = False
        # re-seed the bounded shoulder_pan target from the (reset) joint state
        self._shoulder_pan_target = self._read_shoulder_pan()

    def _read_shoulder_pan(self) -> float:
        """Current shoulder_pan joint angle (rad), clamped to its limits."""
        now = float(self.robot_asset.data.joint_pos[0, self._shoulder_pan_joint_idx].item())
        return float(np.clip(now, self._shoulder_pan_min, self._shoulder_pan_max))

    def _add_device_control_description(self) -> None:
        self._display_controls_table.add_row(["Headset blue button", "start WebXR hand tracking"])
        self._display_controls_table.add_row(["Move right hand up/down/fwd/back", "move gripper via IK (relative)"])
        self._display_controls_table.add_row(["Move right hand left/right", "rotate shoulder_pan joint (relative)"])
        self._display_controls_table.add_row(["Rotate right wrist", "rotate gripper via IK (relative)"])
        self._display_controls_table.add_row(["Pinch thumb+index", "close gripper"])

    def _world_to_root(self, vec_world: np.ndarray) -> np.ndarray:
        """Rotate a world-frame (Isaac) vector into the robot base/root frame."""
        root_quat = self.robot_asset.data.root_quat_w  # (num_envs, 4) wxyz
        vec = torch.tensor(vec_world, device=self.env.device, dtype=torch.float32).unsqueeze(0)
        vec_root = math_utils.quat_apply(math_utils.quat_conjugate(root_quat[:1]), vec)
        return vec_root.squeeze(0).cpu().numpy()

    def _read_root_rot(self) -> Rotation:
        """Current robot base/root orientation in the Isaac world frame (scipy)."""
        rw = self.robot_asset.data.root_quat_w[0]  # (4,) wxyz
        return Rotation.from_quat(rw[[1, 2, 3, 0]].cpu().numpy())  # -> xyzw

    def _read_ee_world_rot(self) -> Rotation:
        """Current gripper (EE) orientation in the Isaac world frame (scipy)."""
        bw = self.robot_asset.data.body_quat_w[0, self._gripper_body_idx]  # (4,) wxyz
        return Rotation.from_quat(bw[[1, 2, 3, 0]].cpu().numpy())  # -> xyzw

    def _anchored_rotation_error(self, rot_now: Rotation) -> np.ndarray:
        """Root-frame rotvec the relative-IK term should apply this step to make the
        gripper chase the hand's absolute orientation (anchored, rate-limited).

        The IK's relative mode applies the command as ``ee_quat_des = dQ @ ee_quat``
        (a world/root-frame left-multiplication), so feeding the **error rotvec**
        ``rotvec(Q_target_root @ Q_ee_root.inv())`` makes ``ee_quat_des = Q_target``
        (one ``max_rot_step_rad``-limited step toward it) and the IK chases the target
        every frame — absolute tracking, drift-free, with no delta accumulation.

        The target is the gripper's anchor orientation composed with the hand's
        rotation-since-anchor, so the gripper mirrors the hand's rotation from its
        start pose. The anchor's gripper orientation (``_anchor_ee_world_rot``) is
        captured as the *current* EE orientation on the same first frame as
        ``_anchor_wrist_rot``, so the hand rotation reference is synchronized with
        the robot at frame 1: the hand has not rotated yet => target == current EE
        => error 0 (no start jump). Any constant hand<->gripper offset is absorbed.

        Axis handling — three corrections, each fixing one cross-axis bleed:
          * **hand heading stripped (swing-twist).** The command is the axis-angle of
            the hand's rotation-since-anchor, and the axis-angle of a compound
            rotation does not separate cleanly: a 90° hand *yaw* rotates a forward
            *roll* into the right axis. A gimbal-free swing-twist split about vertical
            removes the heading twist (owned by shoulder_pan), leaving the hand's
            pitch+roll, so a roll stays a roll at any hand yaw. Done before the swap.
          * **forward<->right swap** is applied to the COMMANDED hand rotation
            (feedforward), not to the error: rotating the hand about its forward axis
            was turning the gripper about right. Permuting the closed-loop *error*
            instead makes the IK chase a swapped direction and one mode diverges;
            doing it on the command keeps the error honest and the loop stable.
          * **target re-headed to the gripper's current heading.** w_cmd is
            calibrated in the root frame; left-multiplying it onto the anchor
            orientation (as a naive version does) leaves a large heading error in the
            chase rotvec once shoulder_pan pans the arm, and that error bleeds a
            forward hand rotation into the gripper's right axis. Rotating the target
            by the pan-since-anchor ``dpsi`` about root up (``dpsi`` = azimuth of the
            gripper forward axis now minus at the anchor) cancels it. No-op at the
            anchor, so the tuned start is unchanged.
          * **the gripper's own up axis is freed, not root up.** Heading is owned by
            shoulder_pan, so the DOF we must not impose is the gripper-up axis
            (body +x). When the wrist is pitched this tilts away from root up, so
            zeroing root-Z (a naive version) drops the wrong component and bleeds a
            forward roll into pitch. We project the gripper-up component out of the
            error instead. Dropping a DOF this way is stable (unlike swapping); we
            are not asking the solver to reach anything.

        All three corrections were verified in a standalone numerical simulation of
        this function: a single-axis hand rotation (roll / pitch) produces a single
        constant-axis gripper rotation at every hand yaw and every gripper pan/pitch,
        and a hand yaw produces no gripper rotation. The naive versions bled roll
        into right (hand yaw, and gripper pan) and roll into pitch (gripper pitch).
        """
        Q_root = self._read_root_rot()
        # Hand rotation since the anchor, mapped through _R_rot_map into the robot
        # world frame.
        dQ_hand = rot_now * self._anchor_wrist_rot.inv()
        R_hand = self._R_rot_map * dQ_hand * self._R_rot_map.inv()
        # Strip the hand's HEADING (its rotation about vertical) before reading the
        # command. The command is the axis-angle of this rotation, and the axis-angle
        # of a compound rotation does NOT separate into clean components: a 90° hand
        # yaw rotates a forward roll into the right axis (visible in the calibration
        # monitor as the "right" value moving when the hand only rolls about the
        # middle-finger axis). A gimbal-free swing–twist split about vertical removes
        # the heading twist, leaving the hand's pitch+roll (tilt) — which is what we
        # mirror; heading is owned by shoulder_pan. Verified in simulation: a forward
        # roll then maps to a fixed command direction at any hand yaw and pitch.
        q = R_hand.as_quat()                          # [x, y, z, w]
        twist_z = np.array([0.0, 0.0, q[2], q[3]])    # projection onto the vertical axis
        n = float(np.linalg.norm(twist_z))
        if n > 1e-8:
            R_hand = Rotation.from_quat(twist_z / n).inv() * R_hand   # remove heading
        w_world = R_hand.as_rotvec()
        w_root = Q_root.inv().apply(w_world)
        # Feedforward forward<->right swap. The component that drives rotation about
        # the RIGHT axis (index 0 after the swap) is sign-inverted — that rotation
        # came out reversed in teleop. Up is left out of the command (it is not
        # controlled — see the error zeroing below).
        w_cmd = np.array([-w_root[1], w_root[0], 0.0])
        Q_ee_root = Q_root.inv() * self._read_ee_world_rot()
        Q_ee_anchor_root = Q_root.inv() * self._anchor_ee_world_rot
        # Two corrections (both verified by numerical simulation) remove the
        # cross-axis bleed that appears once the arm has panned (shoulder_pan) and/or
        # the wrist has pitched. Without them a hand rotation about a single axis
        # turns the gripper about a mix of axes; with them it is a pure single-axis
        # gripper rotation at any pan and pitch.
        #
        # (1) Re-head the target to the gripper's CURRENT heading. w_cmd is
        #     calibrated in the root frame; left-multiplying it onto the anchor
        #     orientation leaves a large heading (azimuth) term in the chase rotvec
        #     once the arm pans, and that term contaminates the forward/right axes.
        #     Rotating the target by the pan-since-anchor dpsi about root up cancels
        #     it. Heading = azimuth of the gripper forward axis (body -z) in root.
        f_now = Q_ee_root.apply(np.array([0.0, 0.0, -1.0]))
        f_anc = Q_ee_anchor_root.apply(np.array([0.0, 0.0, -1.0]))
        dpsi = float(np.arctan2(f_now[1], f_now[0]) - np.arctan2(f_anc[1], f_anc[0]))
        Q_target_root = (
            Rotation.from_rotvec([0.0, 0.0, dpsi])
            * Rotation.from_rotvec(w_cmd)
            * Q_ee_anchor_root
        )
        drot_root = (Q_target_root * Q_ee_root.inv()).as_rotvec()
        # (2) Free the gripper's OWN up axis (body +x), not root up. Heading is owned
        #     by shoulder_pan; the DOF we must not impose is the gripper-up axis.
        #     When the wrist is pitched it tilts away from root up, so zeroing root-Z
        #     (the old code) drops the wrong component and bleeds a forward roll into
        #     pitch. Project the gripper-up component out of the error instead.
        up_axis = Q_ee_root.apply(np.array([1.0, 0.0, 0.0]))
        drot_root = drot_root - np.dot(drot_root, up_axis) * up_axis
        # Soften orientation tracking so it yields to translation: scaling the error
        # down de-weights the rotation rows in the DLS least-squares, so the shared
        # arm joints spend more of their motion on position. Orientation still
        # converges once you stop translating (P-controller, gain < 1).
        drot_root = drot_root * self.rot_track_gain
        # Rate-limit the chase (the IK takes one step per frame toward the target).
        ang = float(np.linalg.norm(drot_root))
        if ang > self.max_rot_step_rad:
            drot_root = drot_root / ang * self.max_rot_step_rad
        return drot_root

    def get_device_state(self) -> np.ndarray:
        """Return the 8D delta: [dx, 0, dz, drvx, drvy, drvz, d_shoulder_pan, d_gripper].

        The IK position/rotation deltas (indices 0:6) are in the robot base/root
        frame; lateral (root-Y) is zeroed there and re-routed to the shoulder_pan
        joint (index 6); index 7 is the relative gripper command.
        """
        wrist_pos, wrist_quat, joints, age = self._state.get()
        out = np.zeros(8, dtype=np.float64)

        # No stream yet (server up but no frames) or tracking stale -> hold still.
        if age > self.safety_timeout_s:
            self._prev_wrist_pos = None
            self._prev_wrist_rot = None
            self._anchor_wrist_rot = None
            self._anchor_ee_world_rot = None
            return out

        # Pinch -> gripper. Drives a relative joint command each step (sim clamps to limits).
        self._close_gripper = pinch_distance(joints) < self.pinch_threshold_m
        out[7] = -self.gripper_sensitivity if self._close_gripper else self.gripper_sensitivity

        # First valid frame after start/reset: anchor wrist + shoulder_pan, no motion.
        if self._prev_wrist_pos is None:
            self._prev_wrist_pos = wrist_pos.copy()
            self._prev_wrist_rot = Rotation.from_quat(wrist_quat)
            # (re)capture the translation-frame + rotation anchors: this block only
            # runs after init/reset/staleness, all of which null them, so the anchors
            # are genuinely held between resets (and across tracking loss they are
            # re-established here).
            self._anchor_wrist_rot = Rotation.from_quat(wrist_quat)
            self._anchor_ee_world_rot = self._read_ee_world_rot()
            self._shoulder_pan_target = self._read_shoulder_pan()
            return out

        # Per-step hand motion decomposed in the HAND's OWN body frame, from the live
        # wrist quaternion — the same axes as the monitor's FWD/RIGHT/UP triad. The
        # command is "how far the hand moved along its own forward/right/up axes", NOT
        # a component of the wrist position in fixed room axes; so moving the hand
        # forward always means "forward where the hand points". Hand (XR) body axes:
        # +X = right, +Y = up, +Z = back (forward = -Z); rot_now.inv() rotates the
        # room-frame position delta into those hand coordinates [right, up, back].
        rot_now = Rotation.from_quat(wrist_quat)
        delta_room = wrist_pos - self._prev_wrist_pos
        hand_delta = rot_now.inv().apply(delta_room) * self.pos_scale      # [right, up, back]
        hand_delta = np.clip(hand_delta, -self.max_pos_step_m, self.max_pos_step_m)
        right_cmd = float(hand_delta[0])     # along hand right -> shoulder_pan
        # fwd_up_gain scales ONLY these two (IK-bound) axes, so forward/back and
        # up/down travel faster per metre of hand motion without touching pan speed.
        up_cmd = float(hand_delta[1]) * self.fwd_up_gain        # along hand up    -> IK (gripper frame)
        fwd_cmd = -float(hand_delta[2]) * self.fwd_up_gain      # along hand fwd (=-back) -> IK (gripper frame)

        # Absolute (anchored) rotation, root frame — see _anchored_rotation_error.
        drot_root = self._anchored_rotation_error(rot_now)

        # Left/right drives shoulder_pan directly, NOT the IK (so101_gamepad_v3
        # convention): integrate into a bounded internal target and emit the relative
        # joint command (target - current). Flip the sign if it pans the wrong way.
        shoulder_pan_now = self._read_shoulder_pan()
        self._shoulder_pan_target += self.shoulder_pan_sensitivity * right_cmd
        self._shoulder_pan_target = float(
            np.clip(self._shoulder_pan_target, self._shoulder_pan_min, self._shoulder_pan_max)
        )

        # Forward/back + up/down are commanded in the GRIPPER body frame and then
        # rotated into the root frame the IK consumes — exactly what so101_gamepad_v3
        # does via _convert_delta_from_frame. The gripper frame moves with the arm, so
        # these stay reachable as the arm pans and translation follows where the
        # gripper points. Gripper-frame axis convention (matches gamepad_v3's delta
        # mapping): forward = -z, up = +x, left/right = y (-> shoulder_pan, 0 here).
        Q_grip_root = self._read_root_rot().inv() * self._read_ee_world_rot()
        v_root = Q_grip_root.apply(np.array([up_cmd, 0.0, -fwd_cmd]))   # gripper [x=up, y=0, z=-fwd]
        out[0] = float(v_root[0])
        out[1] = float(v_root[1])
        out[2] = float(v_root[2])
        out[3:6] = drot_root         # wrist rotation -> IK (absolute, root frame)
        out[6] = self._shoulder_pan_target - shoulder_pan_now  # relative shoulder_pan command

        self._prev_wrist_pos = wrist_pos.copy()
        self._prev_wrist_rot = rot_now
        return out

    # ------------------------------------------------------------------
    # WebXR server (reused verbatim from gregorio/quest3_device.py)
    # ------------------------------------------------------------------
    def _run_server(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        app = web.Application()
        app["state"] = self._state
        app["html"] = self._html
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._cfg_port)
        await site.start()
        print(f"[Quest3] WebSocket server on 0.0.0.0:{self._cfg_port}")
        await asyncio.Event().wait()

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=request.app["html"], content_type="text/html")

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        state: _TeleopState = request.app["state"]
        ws_resp = web.WebSocketResponse()
        await ws_resp.prepare(request)
        print("[Quest3] Browser connected")
        frame_count = 0
        try:
            async for msg in ws_resp:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if "debug" in data:
                            print(f"[quest] {data['debug']}")
                        elif "joints" in data and "wrist_quat" in data:
                            joints = data["joints"]
                            wq = data["wrist_quat"]
                            if len(joints) == 25:
                                pts = np.array(joints, dtype=np.float64)
                                state.update(
                                    wrist_pos=pts[0],
                                    wrist_quat=np.array(wq, dtype=np.float64),
                                    joint_pos=pts,
                                )
                                frame_count += 1
                                if frame_count == 1:
                                    print("[Quest3] First frame received!")
                    except Exception as exc:
                        print(f"[Quest3] Parse error: {exc}")
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            print(f"[Quest3] Browser disconnected after {frame_count} frames")
        return ws_resp
