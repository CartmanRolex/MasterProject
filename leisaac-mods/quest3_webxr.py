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
- ``left_hand_closed`` — left-hand fist detector for the teleop freeze clutch.
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
        self._left_closed = False                            # left-hand clutch (fist) -> freeze
        self._last_update = 0.0

    def update(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        joint_pos: np.ndarray,
        left_closed: bool = False,
    ) -> None:
        with self._lock:
            self._wrist_pos[:] = wrist_pos
            self._wrist_quat[:] = wrist_quat
            self._joint_pos[:] = joint_pos
            self._left_closed = bool(left_closed)
            self._last_update = time.monotonic()

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with self._lock:
            return (
                self._wrist_pos.copy(),
                self._wrist_quat.copy(),
                self._joint_pos.copy(),
                time.monotonic() - self._last_update,
            )

    def left_closed(self) -> bool:
        """Whether the left hand is currently a fist (the teleop clutch is engaged).
        False until the first update, and whenever the left hand is not tracked."""
        with self._lock:
            return self._left_closed


def pinch_distance(joints: np.ndarray) -> float:
    """Thumb-tip <-> index-tip distance (m). joints is the (25, 3) array."""
    return float(np.linalg.norm(joints[THUMB_TIP_IDX] - joints[INDEX_TIP_IDX]))


# Left-hand fingertip / knuckle (MCP) joint indices in the XR 25-joint layout, used
# to detect a closed fist for the teleop clutch. (index, middle, ring, pinky.)
_LEFT_FINGER_TIPS = (9, 14, 19, 24)
_LEFT_FINGER_MCPS = (6, 11, 16, 21)


def left_hand_closed(left_joints, curl_ratio: float = 1.3, min_curled: int = 3) -> bool:
    """True when the left hand is a fist — used as a teleop clutch to freeze commands.

    ``left_joints`` is the left hand's 25 XR joint positions (list/array, metres) or
    ``None`` when the left hand is not tracked. A missing, partial, or malformed hand
    reads as **open** (returns ``False``), so lost left-hand tracking never freezes the
    robot. A finger counts as curled when its tip is no farther from the wrist than
    ``curl_ratio`` times its knuckle (MCP) distance; the hand is closed when at least
    ``min_curled`` of the four non-thumb fingers are curled. The tip/knuckle ratio is
    scale-invariant, so one threshold works across hand sizes.
    """
    if left_joints is None or len(left_joints) != 25:
        return False
    pts = np.asarray(left_joints, dtype=np.float64)
    if pts.shape != (25, 3):
        return False
    wrist = pts[0]
    curled = 0
    for tip, mcp in zip(_LEFT_FINGER_TIPS, _LEFT_FINGER_MCPS):
        tip_d = float(np.linalg.norm(pts[tip] - wrist))
        mcp_d = float(np.linalg.norm(pts[mcp] - wrist))
        if mcp_d > 1e-6 and tip_d < curl_ratio * mcp_d:
            curled += 1
    return curled >= min_curled


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
    anchor_rot: Rotation | None = None,
) -> tuple[np.ndarray, np.ndarray, Rotation]:
    """Per-step EE delta in the Isaac **world** frame, from two wrist poses.

    This is the exact math ``SO101Quest3.get_device_state`` uses (minus the
    final world->robot-root rotation, which is sim-only). Returns
    ``(dpos_world, drotvec_world, rot_now)``; pass ``rot_now`` back as
    ``prev_wrist_rot`` on the next call.

    ``R`` is the XR->Isaac axis map; defaults to the module ``_R_XR_TO_ISAAC``.
    Callers (e.g. the device) may pass their own matrix to tune the axis mapping
    independently — see ``SO101Quest3._R_XR_TO_ISAAC_SO101``.

    ``anchor_rot`` (optional) is a wrist orientation captured at a start/reset
    anchor and held by the caller. When given, the position delta is first
    re-expressed in the hand's *current* body frame — rotated by the inverse of
    the wrist rotation since the anchor — before ``R`` maps it to world. This
    makes the translation follow the hand's pointing direction instead of the
    fixed room axes (which only match the initial hand direction); see the
    device's ``_anchor_wrist_rot``. When ``None`` (e.g. the calibration monitor,
    which does not call this) the legacy room-frame mapping is used unchanged.
    """
    if R is None:
        R = _R_XR_TO_ISAAC

    rot_now = Rotation.from_quat(wrist_quat)

    # Position delta (world/Isaac frame), clamped for safety. When an anchor
    # orientation is given, first re-express the room-frame translation in the
    # hand's current body frame: rotate the delta by the inverse of the wrist
    # rotation since the anchor (dQ = rot_now * anchor_rot.inv()). That makes the
    # translation follow the hand's pointing direction — held until the caller
    # resets the anchor — instead of the fixed room axes, which only match the
    # initial hand direction. With anchor_rot=None this is the legacy room-frame
    # mapping, byte-for-byte unchanged.
    dpos_rm = wrist_pos - prev_wrist_pos
    if anchor_rot is not None:
        dQ = rot_now * anchor_rot.inv()       # wrist rotation since anchor (room frame)
        dpos_rm = dQ.inv().apply(dpos_rm)     # re-express in the hand body frame
    dpos_world = R @ dpos_rm * pos_scale
    dpos_world = np.clip(dpos_world, -max_pos_step_m, max_pos_step_m)

    # Rotation delta as axis-angle (world/Isaac frame), clamped for safety.
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
let lastSend = 0, frameCount = 0, lastFpsStamp = 0, lastValidFrame = 0;
const STALL_MS = 30000;   // end a session whose tracking has been dead this long

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

  // If the headset idles or is taken off, the AR session goes 'hidden' but stays
  // alive with dead tracking, trapping the user in a frozen passthrough with no
  // way to the menus. End it so they drop back to this browser page and can
  // restart. ('visible-blurred' = system menu up; leave that alone.)
  xrSession.addEventListener('visibilitychange', () => {
    if (xrSession && xrSession.visibilityState === 'hidden') {
      sendDebug('Headset idle — ending AR session');
      xrSession.end().catch(() => {});
    }
  });
  lastValidFrame = performance.now();

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

// Watchdog (timer-based, so it keeps running even while the rAF loop is paused
// during idle): if a session is alive but no valid hand frames have arrived for
// STALL_MS, end it so the user is not stuck in dead hand-tracking.
setInterval(() => {
  if (xrSession && performance.now() - lastValidFrame > STALL_MS) {
    sendDebug('Tracking stalled — ending AR session');
    xrSession.end().catch(() => {});
  }
}, 2000);

function onFrame(t, frame) {
  xrSession.requestAnimationFrame(onFrame);

  if (glContext) {
    const layer = xrSession.renderState.baseLayer;
    glContext.bindFramebuffer(glContext.FRAMEBUFFER, layer ? layer.framebuffer : null);
    glContext.clearColor(0, 0, 0, 0);
    glContext.clear(glContext.COLOR_BUFFER_BIT | glContext.DEPTH_BUFFER_BIT);
  }

  if (!refSpace || ws.readyState !== WebSocket.OPEN || t - lastSend < SEND_INTERVAL_MS) return;

  // Gather both hands: the right hand drives teleop (joints + wrist quat); the left
  // hand is the clutch (its joints let the server detect a fist to freeze commands).
  let rightPositions = null, wristQuat = null, leftJoints = null;

  for (const src of frame.session.inputSources) {
    if (!src.hand) continue;

    const positions = [];
    let wq = null;
    let nullCount = 0;

    for (const [jointName, jointSpace] of src.hand) {
      const jp = frame.getJointPose(jointSpace, refSpace);
      if (!jp) { nullCount++; positions.push(null); continue; }
      const p = jp.transform.position;
      positions.push([p.x, p.y, p.z]);
      if (jointName === 'wrist') {
        const o = jp.transform.orientation;
        wq = [o.x, o.y, o.z, o.w];
      }
    }

    const full = (nullCount === 0 && positions.length === 25);
    if (src.handedness === 'right') {
      if (full && wq) { rightPositions = positions; wristQuat = wq; }
    } else if (src.handedness === 'left') {
      // Only a fully-tracked left hand can engage the clutch; a missing or partial
      // left hand leaves leftJoints null (= open), so lost tracking never freezes.
      if (full) leftJoints = positions;
    }
  }

  if (!rightPositions) {
    sendDebug('Partial tracking — keep right hand in view');
    return;
  }

  ws.send(JSON.stringify({ joints: rightPositions, wrist_quat: wristQuat, left_joints: leftJoints }));
  lastSend = t;
  lastValidFrame = performance.now();

  frameCount++;
  if (t - lastFpsStamp >= 1000) {
    status.textContent = 'Sending ' + frameCount + ' fps ✓' + (leftJoints ? ' · L✋' : '');
    fpsEl.textContent = 'wrist: [' + wristQuat.map(v => v.toFixed(3)).join(', ') + ']';
    frameCount = 0; lastFpsStamp = t;
  }
}
</script>
</body>
</html>
"""
