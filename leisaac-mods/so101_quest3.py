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

Tuning: ``pos_scale``, ``rot_scale``, ``gripper_sensitivity``,
``pinch_threshold_m``, and the per-step clamps. The hand->arm axis directions are
set by ``_R_XR_TO_ISAAC_SO101`` (defined in this module and passed to
``xr_delta_to_world`` as ``R``): it maps the XR wrist delta into the Isaac
**world** frame so that ``_world_to_root`` can hand the relative DLS-IK term a
true root-frame delta, and the robot moves up / right / back exactly as the
calibration monitor and ``so101_gamepad_v3`` name those axes. If a motion axis
comes out mirrored or swapped on the robot, flip the corresponding row's sign of
that matrix (or the sign of the relevant scale). It is the device's own copy, so
the calibration monitor (which uses ``quest3_webxr._R_XR_TO_ISAAC``) is unaffected.
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
    xr_delta_to_world,
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
# calibration monitor (its own ``_R_XR_TO_ISAAC``) is unaffected. Flip a row's
# sign if a motion axis comes out mirrored on the real robot.
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
        # also drop the translation-frame anchor so the hand's current pose
        # redefines "forward/right/up" after a reset
        self._anchor_wrist_rot = None
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
            return out

        # Pinch -> gripper. Drives a relative joint command each step (sim clamps to limits).
        self._close_gripper = pinch_distance(joints) < self.pinch_threshold_m
        out[7] = -self.gripper_sensitivity if self._close_gripper else self.gripper_sensitivity

        # First valid frame after start/reset: anchor wrist + shoulder_pan, no motion.
        if self._prev_wrist_pos is None:
            self._prev_wrist_pos = wrist_pos.copy()
            self._prev_wrist_rot = Rotation.from_quat(wrist_quat)
            # (re)capture the translation-frame anchor: this block only runs after
            # init/reset/staleness, all of which null _anchor_wrist_rot, so the
            # anchor is genuinely held between resets (and across tracking loss
            # it is re-established here).
            self._anchor_wrist_rot = Rotation.from_quat(wrist_quat)
            self._shoulder_pan_target = self._read_shoulder_pan()
            return out

        # Per-step EE delta in the Isaac world frame. Uses the device's own axis
        # map (_R_XR_TO_ISAAC_SO101) so the hand→arm directions can be tuned
        # independently of the calibration monitor. ``anchor_rot`` re-expresses the
        # translation in the hand's current body frame so it follows the wrist
        # orientation (held since the last reset).
        dpos_world, drot_world, rot_now = xr_delta_to_world(
            self._prev_wrist_pos,
            self._prev_wrist_rot,
            wrist_pos,
            wrist_quat,
            self.pos_scale,
            self.rot_scale,
            self.max_pos_step_m,
            self.max_rot_step_rad,
            R=_R_XR_TO_ISAAC_SO101,
            anchor_rot=self._anchor_wrist_rot,
        )

        # Rotation axis correction. On this headset the wrist *orientation* axes
        # come in swapped relative to the *translation* axes, so a hand roll about
        # the forward/back axis was driving an EE rotation about the left/right
        # axis. Swap the world X (forward) and Y (left) rotation components so a
        # hand roll about fwd/back maps to an EE roll about fwd/back. In addition,
        # the rotation about the **left/right axis** (EE pitch) comes out inverted
        # on this headset, so negate that component — here the term that lands on
        # the left/right axis is the swapped-forward one (index 0 of the input).
        # Tuning: if a rotation runs the wrong way, flip the sign of the relevant
        # component below; if a *different* axis is the one inverted (e.g. the
        # forward/back roll instead), move the minus sign to the other swapped
        # term — `[drot_world[1], -drot_world[0], drot_world[2]]` — and if a
        # different pair is swapped, change the indices.
        drot_world = np.array([-drot_world[1], drot_world[0], drot_world[2]], dtype=np.float64)

        # World -> robot base frame (the frame the relative DLS-IK term consumes).
        dpos_root = self._world_to_root(dpos_world)
        drot_root = self._world_to_root(drot_world)

        # Lateral (root-Y) is handled by shoulder_pan directly, NOT the IK
        # (so101_gamepad_v3 convention). Integrate it into a bounded internal
        # target and emit the relative joint command (target - current); the
        # RelativeJointPositionAction then drives shoulder_pan to that target.
        shoulder_pan_now = self._read_shoulder_pan()
        self._shoulder_pan_target += self.shoulder_pan_sensitivity * float(dpos_root[1])
        self._shoulder_pan_target = float(
            np.clip(self._shoulder_pan_target, self._shoulder_pan_min, self._shoulder_pan_max)
        )

        out[0] = dpos_root[0]   # forward/back -> IK
        out[1] = 0.0            # lateral removed from IK (driven by shoulder_pan below)
        out[2] = dpos_root[2]   # up/down -> IK
        out[3:6] = drot_root    # wrist rotation -> IK
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
