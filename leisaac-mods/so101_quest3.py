"""Meta Quest 3 hand-tracking teleoperation device for the SO-101 arm.

This reuses the WebXR streaming layer authored by Gregorio for the stock Isaac
Lab Franka demo (``isaac-inference/gregorio/quest3_device.py``): an aiohttp
server serves a WebXR page to the Quest's Meta Browser, the page streams the
right-hand joint poses + wrist quaternion over a WebSocket, and a thumb-index
pinch drives the gripper.

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

Tuning (do this live with the headset): ``pos_scale``, ``rot_scale``,
``gripper_sensitivity``, ``pinch_threshold_m``, and the per-step clamps. If a
motion axis is mirrored, flip the corresponding row of ``_R_XR_TO_ISAAC`` or the
sign of the relevant scale.
"""

import asyncio
import json
import threading
import time

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from aiohttp import WSMsgType, web
from scipy.spatial.transform import Rotation

from .device_base import Device


# WebXR (Y-up, -Z fwd) -> Isaac (Z-up, +X fwd); user stands behind the robot.
# Copied verbatim from gregorio/quest3_device.py.
_R_XR_TO_ISAAC: np.ndarray = np.array(
    [[0, 0, -1],
     [-1, 0, 0],
     [0, 1, 0]],
    dtype=np.float64,
)


class _TeleopState:
    """Written by the WebSocket thread, read by the sim loop. (from gregorio)"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wrist_pos = np.zeros(3, dtype=np.float64)
        self._wrist_quat = np.array([0.0, 0.0, 0.0, 1.0])   # xyzw
        self._joint_pos = np.zeros((25, 3), dtype=np.float64)
        self._last_update = 0.0

    def update(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray, joint_pos: np.ndarray) -> None:
        with self._lock:
            self._wrist_pos[:] = wrist_pos
            self._wrist_quat[:] = wrist_quat
            self._joint_pos[:] = joint_pos
            self._last_update = time.monotonic()

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with self._lock:
            return (
                self._wrist_pos.copy(),
                self._wrist_quat.copy(),
                self._joint_pos.copy(),
                time.monotonic() - self._last_update,
            )


# WebXR client page served to the headset. Copied verbatim from gregorio/quest3_device.py.
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quest3 Isaac Teleop</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif; background: #0f0f1a; color: #e0e0ff;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 100vh; gap: 24px; padding: 20px;
  }
  h1 { font-size: 28px; font-weight: 700; text-align: center; }
  #btn {
    padding: 20px 48px; font-size: 22px; border-radius: 16px;
    background: #5c6bc0; color: #fff; border: none; cursor: pointer;
  }
  #btn:disabled { opacity: 0.4; cursor: default; }
  #status {
    font-size: 14px; opacity: 0.65; background: rgba(255,255,255,0.07);
    padding: 10px 20px; border-radius: 8px; max-width: 380px;
    text-align: center; line-height: 1.5;
  }
  #fps { font-size: 13px; opacity: 0.35; }
</style>
</head>
<body>
  <h1>Quest3 Isaac Teleop</h1>
  <button id="btn">Start Hand Tracking</button>
  <div id="status">Connecting…</div>
  <div id="fps"></div>
<script>
const SEND_INTERVAL_MS = __SEND_INTERVAL_MS__;

const btn    = document.getElementById('btn');
const status = document.getElementById('status');
const fpsEl  = document.getElementById('fps');

const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(wsProto + '//' + location.host + '/ws');
ws.onopen  = () => { status.textContent = 'Server connected — tap to start XR.'; btn.disabled = false; };
ws.onclose = () => { status.textContent = 'Server disconnected.'; btn.disabled = true; };
ws.onerror = () => { status.textContent = 'WebSocket error — check connection.'; };
btn.disabled = true;

function sendDebug(msg) {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ debug: msg }));
  status.textContent = msg;
}

let xrSession = null, refSpace = null, glContext = null;
let lastSend = 0, frameCount = 0, lastFpsStamp = 0;

btn.addEventListener('click', async () => {
  if (!navigator.xr) { sendDebug('ERROR: WebXR not available'); return; }
  const arOk = await navigator.xr.isSessionSupported('immersive-ar');
  if (!arOk) { sendDebug('ERROR: immersive-ar not supported'); return; }

  try {
    xrSession = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['hand-tracking'],
      optionalFeatures: ['local-floor', 'bounded-floor'],
    });
  } catch (e) { sendDebug('ERROR: ' + e.message); return; }

  xrSession.addEventListener('end', () => {
    xrSession = null; refSpace = null;
    btn.disabled = false; btn.textContent = 'Start Hand Tracking';
    sendDebug('Session ended');
  });

  try {
    const canvas = document.createElement('canvas');
    glContext = canvas.getContext('webgl2', { xrCompatible: true, alpha: true })
             || canvas.getContext('webgl',  { xrCompatible: true, alpha: true });
    if (!glContext) { sendDebug('ERROR: WebGL unavailable'); return; }
    await glContext.makeXRCompatible();
    xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, glContext, { alpha: true }) });
  } catch (e) { sendDebug('ERROR base layer: ' + e.message); return; }

  btn.textContent = 'Tracking…'; btn.disabled = true;
  xrSession.requestAnimationFrame(onFrame);

  for (const type of ['local-floor', 'local', 'viewer']) {
    try { refSpace = await xrSession.requestReferenceSpace(type); sendDebug('refSpace: ' + type); break; }
    catch {}
  }
  if (!refSpace) sendDebug('ERROR: no reference space available');
});

setInterval(() => {
  if (!xrSession) return;
  const srcs = Array.from(xrSession.inputSources);
  sendDebug(srcs.length
    ? srcs.map(s => s.handedness + (s.hand ? '[✓]' : '[✗]')).join(', ')
    : 'no input sources yet');
}, 3000);

function onFrame(t, frame) {
  xrSession.requestAnimationFrame(onFrame);

  if (glContext) {
    const layer = xrSession.renderState.baseLayer;
    glContext.bindFramebuffer(glContext.FRAMEBUFFER, layer ? layer.framebuffer : null);
    glContext.clearColor(0, 0, 0, 0);
    glContext.clear(glContext.COLOR_BUFFER_BIT | glContext.DEPTH_BUFFER_BIT);
  }

  if (!refSpace || ws.readyState !== WebSocket.OPEN || t - lastSend < SEND_INTERVAL_MS) return;

  for (const src of frame.session.inputSources) {
    if (src.handedness !== 'right' || !src.hand) continue;

    const positions = [];
    let wristQuat = null;
    let nullCount = 0;

    for (const [jointName, jointSpace] of src.hand) {
      const jp = frame.getJointPose(jointSpace, refSpace);
      if (!jp) { nullCount++; positions.push(null); continue; }
      const p = jp.transform.position;
      positions.push([p.x, p.y, p.z]);
      if (jointName === 'wrist') {
        const o = jp.transform.orientation;
        wristQuat = [o.x, o.y, o.z, o.w];
      }
    }

    if (nullCount > 0 || positions.length !== 25 || !wristQuat) {
      sendDebug('Partial tracking — keep right hand in view (' + nullCount + ' null)');
      break;
    }

    ws.send(JSON.stringify({ joints: positions, wrist_quat: wristQuat }));
    lastSend = t;

    frameCount++;
    if (t - lastFpsStamp >= 1000) {
      status.textContent = 'Sending ' + frameCount + ' fps ✓';
      fpsEl.textContent = 'wrist: [' + wristQuat.map(v => v.toFixed(3)).join(', ') + ']';
      frameCount = 0; lastFpsStamp = t;
    }
    break;
  }
}
</script>
</body>
</html>
"""


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

        # WebXR streaming server (reused from gregorio)
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
        pinch_d = float(np.linalg.norm(joints[4] - joints[9]))  # thumb_tip(4) <-> index_tip(9)
        self._close_gripper = pinch_d < self.pinch_threshold_m
        out[6] = -self.gripper_sensitivity if self._close_gripper else self.gripper_sensitivity

        # First valid frame after start/reset: anchor, no motion this step.
        if self._prev_wrist_pos is None:
            self._prev_wrist_pos = wrist_pos.copy()
            self._prev_wrist_rot = Rotation.from_quat(wrist_quat)
            return out

        # Position delta (world/Isaac frame), clamped for safety.
        dpos_world = _R_XR_TO_ISAAC @ (wrist_pos - self._prev_wrist_pos) * self.pos_scale
        dpos_world = np.clip(dpos_world, -self.max_pos_step_m, self.max_pos_step_m)

        # Rotation delta as axis-angle (world/Isaac frame), clamped for safety.
        rot_now = Rotation.from_quat(wrist_quat)
        rotvec_xr = (rot_now * self._prev_wrist_rot.inv()).as_rotvec()
        drot_world = _R_XR_TO_ISAAC @ rotvec_xr * self.rot_scale
        ang = float(np.linalg.norm(drot_world))
        if ang > self.max_rot_step_rad:
            drot_world = drot_world / ang * self.max_rot_step_rad

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
