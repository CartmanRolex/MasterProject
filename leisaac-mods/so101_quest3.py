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

    joint_state = [dx, dy, dz, drvx, drvy, drvz, d_gripper]   # 7D, root frame

The 6D EE delta is handed to a DLS IK controller over all 5 arm joints
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll); the solver
coordinates the joints to follow the hand, so there is no manual lateral->pan /
roll decomposition like the gamepad device needs. ``d_gripper`` is a relative
joint command (pinch closes).

Start/reset use leisaac's standard desktop keyboard: B starts control, R resets
(fail), N resets (success). The in-headset blue button only starts the WebXR
hand-tracking session.

Tuning: ``pos_scale``, ``rot_scale``, ``gripper_sensitivity``,
``pinch_threshold_m``, and the per-step clamps. If a motion axis is mirrored or
swapped, edit the device axis map ``_R_XR_TO_ISAAC_SO101`` (defined in this
module and passed to ``xr_delta_to_world`` as ``R``) or flip the sign of the
relevant scale. That matrix is the device's own copy so the calibration monitor
(which still uses ``quest3_webxr._R_XR_TO_ISAAC``) is unaffected for now.
"""

import asyncio
import json
import threading

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from aiohttp import WSMsgType, web
from scipy.spatial.transform import Rotation

from .device_base import Device
from .quest3_webxr import (
    _HTML_TEMPLATE,
    _R_XR_TO_ISAAC,  # noqa: F401  (re-exported for callers/tools that import it from here)
    _TeleopState,
    pinch_distance,
    xr_delta_to_world,
)


# Per-step XR->IK axis map used by the *device* (passed to ``xr_delta_to_world``
# as ``R``). The XR wrist axes (x=right, y=up, z=back) line up directly with the
# SO-101 relative-IK position-delta indices in the so101_gamepad_v3 convention —
# idx0=right, idx1=up, idx2=back (forward = -idx2) — so this is the identity.
# Tuned here (NOT in quest3_webxr._R_XR_TO_ISAAC) so the calibration monitor is
# unaffected for now; the two will be reconciled when the monitor is revisited.
# Flip a row / sign if a motion axis comes out mirrored on the real robot.
_R_XR_TO_ISAAC_SO101: np.ndarray = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


class SO101Quest3(Device):
    """Quest 3 hand tracking -> SO-101 relative IK teleoperation.

    ``get_device_state`` returns a 7D root-frame delta:
    ``[dx, dy, dz, drvx, drvy, drvz, d_gripper]``.
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
    ) -> None:
        # Per-step scales. sensitivity is a global multiplier (matches gamepad devices).
        self.pos_scale = 1.0 * sensitivity
        self.rot_scale = 1.0 * sensitivity
        self.gripper_sensitivity = gripper_sensitivity * sensitivity
        self.pinch_threshold_m = pinch_threshold_m
        self.safety_timeout_s = safety_timeout_s
        self.max_pos_step_m = max_pos_step_m
        self.max_rot_step_rad = max_rot_step_rad

        super().__init__(env, "quest3")

        # robot handle (root frame for delta conversion)
        self.robot_asset = self.env.scene["robot"]

        # previous wrist pose, captured on first valid frame (and after every reset)
        self._prev_wrist_pos: np.ndarray | None = None
        self._prev_wrist_rot: Rotation | None = None
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
        self._close_gripper = False

    def _add_device_control_description(self) -> None:
        self._display_controls_table.add_row(["Headset blue button", "start WebXR hand tracking"])
        self._display_controls_table.add_row(["Move right hand", "move gripper (relative)"])
        self._display_controls_table.add_row(["Rotate right wrist", "rotate gripper (relative)"])
        self._display_controls_table.add_row(["Pinch thumb+index", "close gripper"])

    def _world_to_root(self, vec_world: np.ndarray) -> np.ndarray:
        """Rotate a world-frame (Isaac) vector into the robot base/root frame."""
        root_quat = self.robot_asset.data.root_quat_w  # (num_envs, 4) wxyz
        vec = torch.tensor(vec_world, device=self.env.device, dtype=torch.float32).unsqueeze(0)
        vec_root = math_utils.quat_apply(math_utils.quat_conjugate(root_quat[:1]), vec)
        return vec_root.squeeze(0).cpu().numpy()

    def get_device_state(self) -> np.ndarray:
        """Return the 7D root-frame delta for the relative IK + gripper action."""
        wrist_pos, wrist_quat, joints, age = self._state.get()
        out = np.zeros(7, dtype=np.float64)

        # No stream yet (server up but no frames) or tracking stale -> hold still.
        if age > self.safety_timeout_s:
            self._prev_wrist_pos = None
            self._prev_wrist_rot = None
            return out

        # Pinch -> gripper. Drives a relative joint command each step (sim clamps to limits).
        self._close_gripper = pinch_distance(joints) < self.pinch_threshold_m
        out[6] = -self.gripper_sensitivity if self._close_gripper else self.gripper_sensitivity

        # First valid frame after start/reset: anchor, no motion this step.
        if self._prev_wrist_pos is None:
            self._prev_wrist_pos = wrist_pos.copy()
            self._prev_wrist_rot = Rotation.from_quat(wrist_quat)
            return out

        # Per-step EE delta in the Isaac world frame. Uses the device's own axis
        # map (_R_XR_TO_ISAAC_SO101) so the hand→arm directions can be tuned
        # independently of the calibration monitor.
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
        )

        # World -> robot base frame (the frame the relative DLS-IK term consumes).
        out[0:3] = self._world_to_root(dpos_world)
        out[3:6] = self._world_to_root(drot_world)

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
