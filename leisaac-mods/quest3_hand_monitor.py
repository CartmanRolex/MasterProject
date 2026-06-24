#!/usr/bin/env python3
"""Standalone Quest 3 hand-tracking calibration monitor (no Isaac required).

Runs the *same* WebXR server the live ``SO101Quest3`` device uses (it imports the
page, frame matrix and state from ``quest3_webxr.py``), but instead of driving a
robot it serves a self-contained browser **3D calibration dashboard** for the
hand's orientation/coordinate frame — the part of the Quest 3 mapping that is
hardest to get right. One wrist-centered 3D view (mouse-drag to orbit) plots the
hand skeleton plus the wrist's body-frame triad labelled **FWD / RIGHT / UP** (per
the device's room-axis convention: body +X=RIGHT, +Y=UP, +Z=BACK => FWD=-Z), with a
room-frame gizmo for comparison. The side panel reports position (from an origin
anchored on connect) and rotation about **FWD / RIGHT / UP** measured from a
**leveled reference** — the hand's heading at reset with its forward/right axes
forced into the horizontal plane (pitch & roll removed) — plus the raw wrist
quaternion (absolute and since-level). So a flat hand reads zero and the readout
shows tilt away from horizontal. Reset origin (or reconnect) re-anchors the
position origin and re-levels the rotation reference to the current heading. Use it to verify the axis assignment visually — if a triad
arrow does not line up with the physical hand direction, the device's
``_R_rot_map`` needs fixing — without launching the sim.

The axis mapping (``_R_XR_TO_ISAAC`` in ``quest3_webxr.py``) and pinch threshold
are the monitor's; the live device has its own ``_R_XR_TO_ISAAC_SO101`` and is
not affected by changes here.

Run from this directory (so ``import quest3_webxr`` resolves)::

    cd ~/Documents/MasterProject/leisaac-mods
    python quest3_hand_monitor.py            # serves on 0.0.0.0:8080

Connect the Quest exactly like the device (WebXR needs an HTTPS secure context):

    # lowest latency, USB-tethered, no cloud relay:
    adb reverse tcp:8080 tcp:8080            # then open http://localhost:8080 in the Quest browser
    # or, wireless:
    ngrok http 8080                          # then open the https://….ngrok-free.app URL

Then open the dashboard locally in any browser:  http://localhost:8080/monitor
"""

import argparse
import asyncio
import json
import weakref

import numpy as np
from aiohttp import WSMsgType, web
from scipy.spatial.transform import Rotation

from quest3_webxr import (
    _HTML_TEMPLATE,
    _R_XR_TO_ISAAC,
    _TeleopState,
    pinch_distance,
)


# WebXR hand joint order (XRHand iteration order): wrist, then 5 fingers x
# (metacarpal, proximal, [intermediate], distal, tip). Bones for drawing.
_FINGER_STARTS = (1, 5, 10, 15, 20)        # metacarpal index of thumb..pinky
_FINGER_LENS = (4, 5, 5, 5, 5)             # joints per finger (thumb has 4)
_BONES: list[tuple[int, int]] = []
for _start, _ln in zip(_FINGER_STARTS, _FINGER_LENS):
    _BONES.append((0, _start))             # wrist -> metacarpal
    for _j in range(_start, _start + _ln - 1):
        _BONES.append((_j, _j + 1))        # finger chain


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quest3 Hand Monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: ui-monospace, monospace; background: #0f0f1a; color: #e0e0ff;
         display: flex; flex-direction: column; height: 100vh; padding: 12px; gap: 10px; }
  h1 { font-size: 17px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
  #conn { font-size: 12px; opacity: 0.6; font-weight: 400; }
  button { font-family: inherit; font-size: 12px; background: #2a2a45; color: #cfd2ff;
           border: 1px solid #3a3a5a; border-radius: 7px; padding: 6px 12px; cursor: pointer; }
  button:hover { background: #34345a; }
  #wrap { display: flex; gap: 12px; flex: 1; min-height: 0; }
  .panel { background: #16162a; border-radius: 10px; padding: 8px; display: flex;
           flex-direction: column; gap: 6px; flex: 2; min-width: 0; }
  .panel h2 { font-size: 11px; opacity: 0.6; font-weight: 600; }
  canvas { background: #0a0a14; border-radius: 6px; width: 100%; height: 100%;
           display: block; flex: 1; min-height: 0; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #side { flex: 1; display: flex; flex-direction: column; gap: 9px; min-width: 250px; overflow-y: auto; }
  .stat { background: #16162a; border-radius: 10px; padding: 9px 12px; }
  .stat .lbl { font-size: 11px; opacity: 0.55; text-transform: uppercase; letter-spacing: 0.5px; }
  .row { display: flex; justify-content: space-between; font-size: 13px; margin-top: 4px; }
  .row b { color: #9fa8da; }
  #grip { font-size: 22px; font-weight: 800; text-align: center; padding: 10px;
          border-radius: 10px; transition: background 0.05s; }
  .open   { background: #1b3a1b; color: #7ee787; }
  .closed { background: #4a1414; color: #ff8b8b; }
  .note { font-size: 11px; opacity: 0.5; line-height: 1.5; }
  .fwd { color: #ff8a8a; } .right { color: #7ee787; } .up { color: #6bb8ff; }
</style>
</head>
<body>
  <h1>Quest3 Hand Monitor <span id="conn">connecting…</span>
      <button id="reset">Reset origin</button></h1>
  <div id="wrap">
    <div class="panel">
      <h2>RIGHT HAND 3D · skeleton + wrist frame (FWD/RIGHT/UP) · drag to orbit</h2>
      <canvas id="c3d"></canvas>
    </div>
    <div id="side">
      <div id="grip" class="open">GRIPPER</div>
      <div class="stat">
        <div class="lbl">Pinch (thumb–index)</div>
        <div class="row"><span>distance</span><b id="pd">– m</b></div>
        <div class="row"><span>threshold</span><b id="pt">– m</b></div>
      </div>
      <div class="stat">
        <div class="lbl">position — from origin (m)</div>
        <div class="row"><span class="fwd">fwd (+fwd / −back)</span><b id="pf">–</b></div>
        <div class="row"><span class="right">right (+right / −left)</span><b id="pr">–</b></div>
        <div class="row"><span class="up">up (+up / −down)</span><b id="pu">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">rotation — from level (deg)</div>
        <div class="row"><span class="fwd">about fwd</span><b id="rf">–</b></div>
        <div class="row"><span class="right">about right</span><b id="rr">–</b></div>
        <div class="row"><span class="up">about up</span><b id="ru">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">wrist quaternion (w, x, y, z)</div>
        <div class="row"><span>absolute</span><b id="qnow">–</b></div>
        <div class="row"><span>since level</span><b id="qdel">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">stream</div>
        <div class="row"><span>fps</span><b id="fps">–</b></div>
        <div class="row"><span>wrist pos (xr, m)</span><b id="wp">–</b></div>
      </div>
      <div class="stat note">
        3D view is centered on the wrist (hand stays visible while you rotate it).
        Arrows = the wrist body axes labeled per the device's room-axis convention
        (<span class="fwd">FWD</span>=body −Z,
        <span class="right">RIGHT</span>=body +X,
        <span class="up">UP</span>=body +Y). If an arrow does not line up with the
        physical hand direction, that axis assignment is wrong — fix it in the
        device's <code>_R_rot_map</code>. Corner gizmo shows the room frame
        (R/U/F) under the same camera for comparison. Position is relative to the
        origin (set on connect / by <b>Reset origin</b>); rotation is measured from a
        leveled reference — the hand's heading at reset with forward/right forced
        into the horizontal plane — so a flat hand reads zero and it shows tilt.
      </div>
    </div>
  </div>
<script>
const BONES = __BONES__;
const TRAIL_MAX = 120;
const $ = id => document.getElementById(id);
const conn = $('conn');
let trail = [];          // recent wrist positions (room frame) for the motion path
let last = {};

function fitCanvas(c) {
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, r.width | 0); c.height = Math.max(1, r.height | 0);
}
const cv = $('c3d'), g = cv.getContext('2d');
function fitAll() { fitCanvas(cv); }
window.addEventListener('resize', fitAll); fitAll();

// ---- hand-rolled 3D: orbit camera (az around up, el around right) + orthographic ----
let az = -0.7, el = 0.45;          // initial camera angle (radians)
let dragging = false, lx = 0, ly = 0;
cv.addEventListener('mousedown', e => { dragging = true; lx = e.clientX; ly = e.clientY; });
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  az += (e.clientX - lx) * 0.01;
  el = Math.max(-1.5, Math.min(1.5, el - (e.clientY - ly) * 0.01));
  lx = e.clientX; ly = e.clientY; draw3D();
});
window.addEventListener('mouseup', () => dragging = false);

function rotateCam(p) {           // room point -> camera space
  let x = p[0], y = p[1], z = p[2];
  const ca = Math.cos(az), sa = Math.sin(az);
  let x1 = ca * x + sa * z, z1 = -sa * x + ca * z;   // yaw about up
  const ce = Math.cos(el), se = Math.sin(el);
  let y2 = ce * y - se * z1, z2 = se * y + ce * z1;  // pitch about right
  return [x1, y2, z2];
}

function draw3D() {
  const w = cv.width, h = cv.height, cx = w / 2, cy = h / 2;
  g.clearRect(0, 0, w, h);
  const joints = last.joints, wp = last.wrist_pos;
  if (!joints || joints.length !== 25 || !wp) return;
  // skeleton relative to the wrist (view stays centered on the hand)
  const rel = joints.map(p => [p[0] - wp[0], p[1] - wp[1], p[2] - wp[2]]);
  let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (const p of rel) for (let i = 0; i < 3; i++) { mn[i] = Math.min(mn[i], p[i]); mx[i] = Math.max(mx[i], p[i]); }
  const span = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 0.05);
  const sc = Math.min(w, h) * 0.40 / span;
  const triadLen = span * 0.7;
  const proj = p => { const r = rotateCam(p); return [cx + r[0] * sc, cy - r[1] * sc, r[2]]; };

  // recent wrist motion path (relative to current wrist)
  if (trail.length > 1) {
    g.strokeStyle = '#3949ab'; g.lineWidth = 1.5; g.beginPath();
    for (let i = 0; i < trail.length; i++) {
      const q = proj([trail[i][0]-wp[0], trail[i][1]-wp[1], trail[i][2]-wp[2]]);
      i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]);
    }
    g.stroke();
  }
  // bones
  g.strokeStyle = '#5c6bc0'; g.lineWidth = 2;
  for (const [a, b] of BONES) {
    const pa = proj(rel[a]), pb = proj(rel[b]);
    g.beginPath(); g.moveTo(pa[0], pa[1]); g.lineTo(pb[0], pb[1]); g.stroke();
  }
  // joints
  for (let i = 0; i < 25; i++) {
    const q = proj(rel[i]);
    g.fillStyle = (i === 4 || i === 9) ? '#ffd54f' : (i === 0 ? '#fff' : '#9fa8da');
    g.beginPath(); g.arc(q[0], q[1], i === 0 ? 4 : 2.5, 0, 7); g.fill();
  }
  // wrist body-frame triad (FWD/RIGHT/UP) at center
  if (last.triad) {
    const arrow = (dir, color, lbl) => {
      const tip = proj([dir[0]*triadLen, dir[1]*triadLen, dir[2]*triadLen]);
      g.strokeStyle = color; g.lineWidth = 2.5;
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(tip[0], tip[1]); g.stroke();
      g.fillStyle = color; g.font = '11px ui-monospace, monospace';
      g.fillText(lbl, tip[0] + 3, tip[1] - 3);
    };
    arrow(last.triad.fwd,   '#ff8a8a', 'FWD');
    arrow(last.triad.right, '#7ee787', 'RIGHT');
    arrow(last.triad.up,    '#6bb8ff', 'UP');
  }
  // room-frame gizmo in the corner (R/U/F under the same camera)
  const ox = 44, oy = h - 44, gs = 26;
  const giz = (dir, color, lbl) => {
    const r = rotateCam(dir); const tx = ox + r[0]*gs, ty = oy - r[1]*gs;
    g.strokeStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(ox, oy); g.lineTo(tx, ty); g.stroke();
    g.fillStyle = color; g.font = '9px ui-monospace, monospace'; g.fillText(lbl, tx + 2, ty - 2);
  };
  giz([1,0,0], '#7ee787', 'R'); giz([0,1,0], '#6bb8ff', 'U'); giz([0,0,-1], '#ff8a8a', 'F');
}

const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
let ws;
function connect() {
  ws = new WebSocket(proto + '//' + location.host + '/monitor_ws');
  ws.onopen  = () => conn.textContent = 'connected';
  ws.onclose = () => { conn.textContent = 'disconnected — retrying…'; setTimeout(connect, 1000); };
  ws.onmessage = ev => {
    const d = JSON.parse(ev.data); last = d;
    if (d.wrist_pos) { trail.push(d.wrist_pos); if (trail.length > TRAIL_MAX) trail.shift(); }
    draw3D();
    const f = (v, n=4) => (v >= 0 ? ' ' : '') + v.toFixed(n);
    const f3 = v => (v >= 0 ? ' ' : '') + v.toFixed(3);
    $('pd').textContent = f(d.pinch_d, 4) + ' m';
    $('pt').textContent = f(d.pinch_threshold, 4) + ' m';
    $('pf').textContent = f3(d.pos[0]); $('pr').textContent = f3(d.pos[1]); $('pu').textContent = f3(d.pos[2]);
    $('rf').textContent = f(d.rot[0], 1); $('rr').textContent = f(d.rot[1], 1); $('ru').textContent = f(d.rot[2], 1);
    $('qnow').textContent = d.quat_now.map(v => f(v,3)).join(' ');
    $('qdel').textContent = d.quat_delta.map(v => f(v,3)).join(' ');
    $('fps').textContent = d.fps;
    $('wp').textContent = d.wrist_pos.map(v => v.toFixed(2)).join(', ');
    const grip = $('grip'); grip.textContent = d.gripper.toUpperCase(); grip.className = d.gripper;
  };
}
$('reset').onclick = () => { trail = []; if (ws && ws.readyState === 1) ws.send(JSON.stringify({ cmd: 'reset_origin' })); };
connect();
</script>
</body>
</html>
"""


class Monitor:
    """Receives the Quest stream, computes the device command, fans out to dashboards."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = _TeleopState()
        self.webxr_html = _HTML_TEMPLATE.replace("__SEND_INTERVAL_MS__", str(1000 // args.send_hz))
        self.dashboard_html = (
            _DASHBOARD_HTML
            .replace("__BONES__", json.dumps(_BONES))
        )
        self.monitors: weakref.WeakSet = weakref.WeakSet()
        # Position origin (set on the first frame / when "Reset origin" is pressed),
        # so position is reported as displacement from where the hand is *now*.
        self._origin_pos: np.ndarray | None = None
        # Rotation reference: the hand's heading at reset, *leveled* — its forward and
        # right axes forced into the horizontal plane (pitch & roll removed, up = room
        # +Y). Re-captured on connect / Reset origin from the next frame's pose, so
        # rotation is reported as deviation from that flat, level hand.
        self._origin_rot: Rotation = Rotation.identity()
        # fps accounting
        self._fps = 0
        self._fps_count = 0
        self._fps_t0 = 0.0

    @staticmethod
    def _leveled_reference(rot_now: Rotation) -> Rotation:
        """The hand's current heading, leveled to the horizontal plane.

        Keeps which way the hand points (yaw) but forces its forward and right axes
        into the horizontal plane — pitch and roll removed, up = room +Y. The result
        is the zero reference so the rotation readout shows tilt away from flat. The
        room frame is XR's: +X = right, +Y = up, +Z = back, so forward = -bodyZ.
        """
        R = rot_now.as_matrix()
        fwd = -R[:, 2]                                   # hand forward, room coords
        horiz = np.array([fwd[0], 0.0, fwd[2]])          # project onto the floor plane
        n = np.linalg.norm(horiz)
        if n < 1e-6:                                     # hand pointing straight up/down
            return Rotation.identity()
        fwd_l = horiz / n
        up_l = np.array([0.0, 1.0, 0.0])
        right_l = np.cross(up_l, -fwd_l)                 # X = Y x Z (back = -forward)
        # Columns are the hand body axes [right, up, back] in room coords (det +1).
        return Rotation.from_matrix(np.column_stack([right_l, up_l, -fwd_l]))

    # ---- current pose, mapped into the FWD/RIGHT/UP frame ----
    def compute(self, wrist_pos, wrist_quat, joints) -> dict:
        rot_now = Rotation.from_quat(wrist_quat)
        if self._origin_pos is None:
            self._origin_pos = wrist_pos.copy()
            self._origin_rot = self._leveled_reference(rot_now)
        # Position relative to the origin, mapped into [up, right, back] then
        # relabelled to [fwd, right, up] (forward = -back). Absolute displacement of
        # the hand from the anchor — no clamping / integration.
        m = _R_XR_TO_ISAAC @ (wrist_pos - self._origin_pos) * self.args.pos_scale   # [up,right,back]
        pos = np.array([-m[2], m[1], m[0]])                                          # [fwd,right,up]
        # Rotation relative to the flat/level reference, mapped the same way and
        # relabelled to [about fwd, about right, about up] (deg).
        rotvec_xr = (rot_now * self._origin_rot.inv()).as_rotvec()
        r = _R_XR_TO_ISAAC @ rotvec_xr * self.args.rot_scale                          # [up,right,back]
        rot = np.rad2deg(np.array([-r[2], r[1], r[0]]))                               # [fwd,right,up]
        # Wrist body-frame triad in the ROOM frame, labelled with the device's
        # room-axis convention (body +X = RIGHT, +Y = UP, +Z = BACK => FWD = -Z).
        # Columns of rot_now.as_matrix() are the wrist's local X/Y/Z in room coords.
        # This is the convention the dashboard verifies: if an arrow does not line up
        # with the physical hand, the device's axis assignment is wrong.
        R = rot_now.as_matrix()
        triad = {
            "right": R[:, 0].tolist(),
            "up":    R[:, 1].tolist(),
            "fwd":   (-R[:, 2]).tolist(),
        }
        # Quaternions as (w, x, y, z). scipy stores xyzw.
        qn = rot_now.as_quat(); qd = (rot_now * self._origin_rot.inv()).as_quat()
        quat_now = [qn[3], qn[0], qn[1], qn[2]]
        quat_delta = [qd[3], qd[0], qd[1], qd[2]]
        pd = pinch_distance(joints)
        return {
            "joints": joints.tolist(),
            "wrist_pos": wrist_pos.tolist(),
            "pos": pos.tolist(),
            "rot": rot.tolist(),
            "triad": triad,
            "quat_now": quat_now,
            "quat_delta": quat_delta,
            "pinch_d": pd,
            "pinch_threshold": self.args.pinch_threshold,
            "gripper": "closed" if pd < self.args.pinch_threshold else "open",
            "fps": self._fps,
        }

    async def broadcast(self, payload: dict) -> None:
        if not self.monitors:
            return
        msg = json.dumps(payload)
        for ws in list(self.monitors):
            try:
                await ws.send_str(msg)
            except Exception:
                pass

    # ---- routes ----
    async def index(self, request: web.Request) -> web.Response:
        return web.Response(text=self.webxr_html, content_type="text/html")

    async def dashboard(self, request: web.Request) -> web.Response:
        return web.Response(text=self.dashboard_html, content_type="text/html")

    async def monitor_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.monitors.add(ws)
        print("[Monitor] Dashboard connected")
        try:
            async for msg in ws:  # dashboard sends only the occasional reset command
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    cmd = json.loads(msg.data)
                except Exception:
                    continue
                if cmd.get("cmd") == "reset_origin":
                    # Next frame re-anchors position and re-levels the rotation
                    # reference to the hand's current (leveled) heading.
                    self._origin_pos = None
        finally:
            print("[Monitor] Dashboard disconnected")
        return ws

    async def quest_ws(self, request: web.Request) -> web.WebSocketResponse:
        import time

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("[Monitor] Quest connected")
        # fresh position anchor for this session; rotation stays referenced to level
        self._origin_pos = None
        self._origin_rot = Rotation.identity()
        self._fps_t0 = time.monotonic()
        self._fps_count = 0
        first = True
        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    if m.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                        break
                    continue
                try:
                    data = json.loads(m.data)
                except Exception:
                    continue
                if "debug" in data:
                    print(f"[quest] {data['debug']}")
                    continue
                if "joints" not in data or "wrist_quat" not in data:
                    continue
                if len(data["joints"]) != 25:
                    continue
                if first:
                    print("[Monitor] First frame received!")
                    first = False
                joints = np.array(data["joints"], dtype=np.float64)
                wq = np.array(data["wrist_quat"], dtype=np.float64)
                self.state.update(wrist_pos=joints[0], wrist_quat=wq, joint_pos=joints)
                # fps
                self._fps_count += 1
                now = time.monotonic()
                if now - self._fps_t0 >= 1.0:
                    self._fps = self._fps_count
                    self._fps_count = 0
                    self._fps_t0 = now
                payload = self.compute(joints[0], wq, joints)
                await self.broadcast(payload)
        finally:
            print("[Monitor] Quest disconnected")
        return ws

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.quest_ws)
        app.router.add_get("/monitor", self.dashboard)
        app.router.add_get("/monitor_ws", self.monitor_ws)
        return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--send-hz", type=int, default=60, help="WebXR page send rate (matches device default)")
    # device defaults (SO101Quest3.__init__) so calibration transfers 1:1
    p.add_argument("--pos-scale", type=float, default=1.0)
    p.add_argument("--rot-scale", type=float, default=1.0)
    p.add_argument("--pinch-threshold", type=float, default=0.035, help="metres")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    monitor = Monitor(args)
    app = monitor.build_app()
    print(f"[Monitor] WebXR page  : http://localhost:{args.port}/         (open in the Quest browser)")
    print(f"[Monitor] Dashboard   : http://localhost:{args.port}/monitor  (open on this computer)")
    print(f"[Monitor] Tunnel the Quest with `adb reverse tcp:{args.port} tcp:{args.port}` (USB) or `ngrok http {args.port}`.")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
