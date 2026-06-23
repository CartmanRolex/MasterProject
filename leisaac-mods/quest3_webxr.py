"""Shared, Isaac-free WebXR layer for Quest 3 hand teleoperation.

This module holds the pieces that must stay **identical** between the live
teleop device (``so101_quest3.py``, which runs inside Isaac Sim) and the
standalone calibration tool (``quest3_hand_monitor.py``, which has no Isaac
dependency). Keeping a single source of truth means anything you tune here —
``_R_XR_TO_ISAAC``, the scales, the per-step clamps, the pinch indices — applies
to both, so calibration done with the monitor transfers to the real device with
no extra steps.

Only ``numpy`` + ``scipy`` + ``aiohttp`` are imported (no ``torch`` / Isaac), so
this file can be imported and run on a laptop without the Isaac environment.

Contents:
- ``_HTML_TEMPLATE``  — the WebXR page served to the Quest's Meta Browser.
- ``_R_XR_TO_ISAAC``  — WebXR (Y-up, -Z fwd) -> Isaac (Z-up, +X fwd) rotation.
- ``_TeleopState``    — thread-safe latch written by the WebSocket, read by sim.
- ``THUMB_TIP_IDX`` / ``INDEX_TIP_IDX`` / ``pinch_distance`` — gripper pinch.
- ``xr_delta_to_world`` — per-step EE delta in the Isaac **world** frame.
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


# XR wrist axes -> robot display frame, used by the calibration monitor only
# (the live device passes its own ``_R_XR_TO_ISAAC_SO101``). The XR wrist position
# components are [right, up, back]; the robot display frame is the gamepad_v3
# convention [idx0 = up, idx1 = right (positive = to the right), idx2 = back
# (forward = -idx2)]. So this swaps right<->up and keeps back:
#     idx0 (up)    <- wrist[1]
#     idx1 (right) <- wrist[0]
#     idx2 (back)  <- wrist[2]
# (det = -1: this is a reflection, fine for the monitor's position/orientation
# display.) Flip a row's sign if an axis comes out mirrored after repositioning
# the headset.
_R_XR_TO_ISAAC: np.ndarray = np.array(
    [[0, 1, 0],
     [1, 0, 0],
     [0, 0, 1]],
    dtype=np.float64,
)

# WebXR hand joint indices used for the pinch->gripper signal.
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 9


class _TeleopState:
    """Written by the WebSocket thread, read by the sim/monitor loop. (from gregorio)"""

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


def pinch_distance(joints: np.ndarray) -> float:
    """Thumb-tip <-> index-tip distance (m). joints is the (25, 3) array."""
    return float(np.linalg.norm(joints[THUMB_TIP_IDX] - joints[INDEX_TIP_IDX]))


def xr_delta_to_world(
    prev_wrist_pos: np.ndarray,
    prev_wrist_rot: Rotation,
    wrist_pos: np.ndarray,
    wrist_quat: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    max_pos_step_m: float,
    max_rot_step_rad: float,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Rotation]:
    """Per-step EE delta in the Isaac **world** frame, from two wrist poses.

    This is the exact math ``SO101Quest3.get_device_state`` uses (minus the
    final world->robot-root rotation, which is sim-only). Returns
    ``(dpos_world, drotvec_world, rot_now)``; pass ``rot_now`` back as
    ``prev_wrist_rot`` on the next call.

    ``R`` is the XR->Isaac axis map; defaults to the module ``_R_XR_TO_ISAAC``.
    Callers (e.g. the device) may pass their own matrix to tune the axis mapping
    independently — see ``SO101Quest3._R_XR_TO_ISAAC_SO101``.
    """
    if R is None:
        R = _R_XR_TO_ISAAC

    # Position delta (world/Isaac frame), clamped for safety.
    dpos_world = R @ (wrist_pos - prev_wrist_pos) * pos_scale
    dpos_world = np.clip(dpos_world, -max_pos_step_m, max_pos_step_m)

    # Rotation delta as axis-angle (world/Isaac frame), clamped for safety.
    rot_now = Rotation.from_quat(wrist_quat)
    rotvec_xr = (rot_now * prev_wrist_rot.inv()).as_rotvec()
    drot_world = R @ rotvec_xr * rot_scale
    ang = float(np.linalg.norm(drot_world))
    if ang > max_rot_step_rad:
        drot_world = drot_world / ang * max_rot_step_rad

    return dpos_world, drot_world, rot_now


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
