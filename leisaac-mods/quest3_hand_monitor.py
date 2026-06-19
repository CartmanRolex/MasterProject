#!/usr/bin/env python3
"""Standalone Quest 3 hand-tracking calibration monitor (no Isaac required).

Runs the *same* WebXR server the live ``SO101Quest3`` device uses (it imports the
page, frame matrix, state and delta math from ``quest3_webxr.py``), but instead
of driving a robot it computes the *exact* per-step command the device would emit
and streams it — together with the raw hand skeleton — to a self-contained
browser dashboard. Open the dashboard beside your WebRTC window, place the
headset in different spots, and watch how tracking and the XR->Isaac mapping
respond, without paying the cost of launching Isaac Sim each time.

Because the command is computed with ``xr_delta_to_world`` / ``pinch_distance``
from ``quest3_webxr.py`` (the same code the device runs), anything you tune here
— ``_R_XR_TO_ISAAC``, the scales, the clamps, ``--pinch-threshold`` — transfers
to the real device unchanged.

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
    xr_delta_to_world,
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
  .views { display: flex; gap: 12px; flex: 2; }
  .panel { background: #16162a; border-radius: 10px; padding: 8px; display: flex;
           flex-direction: column; gap: 6px; flex: 1; min-width: 0; }
  .panel h2 { font-size: 11px; opacity: 0.6; font-weight: 600; }
  canvas { background: #0a0a14; border-radius: 6px; width: 100%; height: 100%;
           display: block; flex: 1; min-height: 0; }
  #side { flex: 1; display: flex; flex-direction: column; gap: 9px; min-width: 250px; overflow-y: auto; }
  .stat { background: #16162a; border-radius: 10px; padding: 9px 12px; }
  .stat .lbl { font-size: 11px; opacity: 0.55; text-transform: uppercase; letter-spacing: 0.5px; }
  .row { display: flex; justify-content: space-between; font-size: 13px; margin-top: 4px; }
  .row b { color: #9fa8da; }
  #grip { font-size: 24px; font-weight: 800; text-align: center; padding: 12px;
          border-radius: 10px; transition: background 0.05s; }
  .open   { background: #1b3a1b; color: #7ee787; }
  .closed { background: #4a1414; color: #ff8b8b; }
  .note { font-size: 11px; opacity: 0.5; line-height: 1.5; }
  .ax0 { color: #ff8a8a; } .ax1 { color: #7ee787; } .ax2 { color: #6bb8ff; }
</style>
</head>
<body>
  <h1>Quest3 Hand Monitor <span id="conn">connecting…</span>
      <button id="reset">Reset EE origin</button></h1>
  <div id="wrap">
    <div class="views">
      <div class="panel"><h2>ROBOT FRAME — TOP · forward × lateral</h2><canvas id="cTop"></canvas></div>
      <div class="panel"><h2>ROBOT FRAME — SIDE · forward × up</h2><canvas id="cSide"></canvas></div>
      <div class="panel"><h2>RIGHT HAND (front) — pinch reference</h2><canvas id="cHand"></canvas></div>
    </div>
    <div id="side">
      <div id="grip" class="open">GRIPPER</div>
      <div class="stat">
        <div class="lbl">Pinch (thumb–index)</div>
        <div class="row"><span>distance</span><b id="pd">– m</b></div>
        <div class="row"><span>threshold</span><b id="pt">– m</b></div>
      </div>
      <div class="stat">
        <div class="lbl">dpos — per step · robot index (gamepad frame)</div>
        <div class="row"><span>idx0 → UP</span><b id="dx">–</b></div>
        <div class="row"><span>idx1 → LATERAL</span><b id="dy">–</b></div>
        <div class="row"><span>idx2 → BACK (−=fwd)</span><b id="dz">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">ee — integrated command (m)</div>
        <div class="row"><span>up / lat / back</span><b id="ee">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">drot — per step · rotvec</div>
        <div class="row"><span>r0 → ROLL</span><b id="rx">–</b></div>
        <div class="row"><span>r1 → PITCH</span><b id="ry">–</b></div>
        <div class="row"><span>r2 → YAW</span><b id="rz">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">stream</div>
        <div class="row"><span>fps</span><b id="fps">–</b></div>
        <div class="row"><span>wrist pos (xr)</span><b id="wp">–</b></div>
      </div>
      <div class="stat note">
        Robot-frame views use the gamepad convention: <b>idx0 = up</b>,
        <b>idx1 = lateral</b>, <b>idx2 = back (forward = −idx2)</b>.
        Yellow dot = integrated EE; green arrow = this step's dpos; triad =
        wrist axes mapped to the robot frame (<span class="ax0">x</span>
        <span class="ax1">y</span> <span class="ax2">z</span>).
      </div>
    </div>
  </div>
<script>
const BONES = __BONES__;
const VIEW_M = __VIEW_M__;          // metres across the fixed-scale robot views
const ARROW_GAIN = 2500;            // px per metre of dpos
const ARROW_MAX = 70, TRIAD_LEN = 46, TRAIL_MAX = 240;
const $ = id => document.getElementById(id);
const conn = $('conn');
let trail = [];
let last = {};

function fitCanvas(c) {
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, r.width | 0); c.height = Math.max(1, r.height | 0);
}
const cTop = $('cTop'), cSide = $('cSide'), cHand = $('cHand');
const gTop = cTop.getContext('2d'), gSide = cSide.getContext('2d'), gHand = cHand.getContext('2d');
function fitAll() { fitCanvas(cTop); fitCanvas(cSide); fitCanvas(cHand); }
window.addEventListener('resize', fitAll); fitAll();

// Robot-frame vector v = [idx0 up, idx1 lateral, idx2 back]; forward = -idx2.
const projTop  = v => [-v[2], v[1]];   // horizontal = forward, vertical = lateral
const projSide = v => [-v[2], v[0]];   // horizontal = forward, vertical = up
const col = (M, k) => [M[0][k], M[1][k], M[2][k]];   // column k of a 3x3

// Fixed-scale robot-frame view: EE trail + dot + dpos arrow + wrist triad.
function drawRobotView(ctx, cv, proj, topLbl, botLbl) {
  const w = cv.width, h = cv.height, cx = w / 2, cy = h / 2;
  const pxm = Math.min(w, h) / VIEW_M;
  const sx = hv => cx + hv[0] * pxm, sy = hv => cy - hv[1] * pxm;
  ctx.clearRect(0, 0, w, h);
  // axes cross + labels
  ctx.strokeStyle = '#23233c'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
  ctx.fillStyle = '#666'; ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('FWD +', w - 44, cy - 6);
  ctx.fillText('BACK', 5, cy - 6);
  ctx.fillText(topLbl, cx + 6, 12);
  ctx.fillText(botLbl, cx + 6, h - 5);
  // EE trail
  if (trail.length > 1) {
    ctx.strokeStyle = '#3949ab'; ctx.lineWidth = 1.5; ctx.beginPath();
    trail.forEach((p, i) => { const q = proj(p); const x = sx(q), y = sy(q); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  const e = last.ee_pos ? proj(last.ee_pos) : [0, 0];
  const ex = sx(e), ey = sy(e);
  // dpos arrow (this step's command direction)
  if (last.dpos) {
    const dp = proj(last.dpos); let ax = dp[0] * ARROW_GAIN, ay = dp[1] * ARROW_GAIN;
    const L = Math.hypot(ax, ay);
    if (L > ARROW_MAX) { ax = ax / L * ARROW_MAX; ay = ay / L * ARROW_MAX; }
    if (L > 0.5) {
      ctx.strokeStyle = '#7ee787'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex + ax, ey - ay); ctx.stroke();
    }
  }
  // wrist orientation triad mapped to robot frame
  if (last.ee_basis) {
    const cols = ['#ff8a8a', '#7ee787', '#6bb8ff'], names = ['x', 'y', 'z'];
    for (let k = 0; k < 3; k++) {
      const pc = proj(col(last.ee_basis, k));
      const tx = ex + pc[0] * TRIAD_LEN, ty = ey - pc[1] * TRIAD_LEN;
      ctx.strokeStyle = cols[k]; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.fillStyle = cols[k]; ctx.fillText(names[k], tx + 2, ty - 2);
    }
  }
  // EE dot on top
  ctx.fillStyle = '#ffd54f'; ctx.beginPath(); ctx.arc(ex, ey, 5, 0, 7); ctx.fill();
}

// Auto-fit hand skeleton (translation-free, by design) for pinch reference.
function drawHand(ctx, cv, joints, pick) {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!joints || joints.length !== 25) return;
  const pts = joints.map(pick);
  let minH = 1e9, maxH = -1e9, minV = 1e9, maxV = -1e9;
  for (const [h, v] of pts) { minH = Math.min(minH, h); maxH = Math.max(maxH, h);
                              minV = Math.min(minV, v); maxV = Math.max(maxV, v); }
  const pad = 24, w = cv.width - 2 * pad, h = cv.height - 2 * pad;
  const span = Math.max(maxH - minH, maxV - minV, 0.05);
  const cH = (minH + maxH) / 2, cV = (minV + maxV) / 2;
  const sc = Math.min(w, h) / span;
  const X = hv => pad + w / 2 + (hv[0] - cH) * sc;
  const Y = hv => pad + h / 2 - (hv[1] - cV) * sc;
  ctx.strokeStyle = '#5c6bc0'; ctx.lineWidth = 2;
  for (const [a, b] of BONES) {
    ctx.beginPath(); ctx.moveTo(X(pts[a]), Y(pts[a])); ctx.lineTo(X(pts[b]), Y(pts[b])); ctx.stroke();
  }
  for (let i = 0; i < 25; i++) {
    ctx.fillStyle = (i === 4 || i === 9) ? '#ffd54f' : (i === 0 ? '#fff' : '#9fa8da');
    ctx.beginPath(); ctx.arc(X(pts[i]), Y(pts[i]), (i === 0 ? 5 : 3), 0, 7); ctx.fill();
  }
}

const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
let ws;
function connect() {
  ws = new WebSocket(proto + '//' + location.host + '/monitor_ws');
  ws.onopen  = () => conn.textContent = 'connected';
  ws.onclose = () => { conn.textContent = 'disconnected — retrying…'; setTimeout(connect, 1000); };
  ws.onmessage = ev => {
    const d = JSON.parse(ev.data); last = d;
    if (d.ee_pos) { trail.push(d.ee_pos); if (trail.length > TRAIL_MAX) trail.shift(); }
    drawRobotView(gTop,  cTop,  projTop,  'LATERAL +', 'LATERAL −');
    drawRobotView(gSide, cSide, projSide, 'UP +', 'DOWN −');
    drawHand(gHand, cHand, d.joints, p => [p[0], p[1]]);   // X, Y
    const f = (v, n=4) => (v >= 0 ? ' ' : '') + v.toFixed(n);
    $('pd').textContent = f(d.pinch_d, 4) + ' m';
    $('pt').textContent = f(d.pinch_threshold, 4) + ' m';
    $('dx').textContent = f(d.dpos[0]); $('dy').textContent = f(d.dpos[1]); $('dz').textContent = f(d.dpos[2]);
    $('rx').textContent = f(d.drot[0]); $('ry').textContent = f(d.drot[1]); $('rz').textContent = f(d.drot[2]);
    $('ee').textContent = d.ee_pos.map(v => v.toFixed(3)).join(', ');
    $('fps').textContent = d.fps;
    $('wp').textContent = d.wrist_pos.map(v => v.toFixed(2)).join(', ');
    const g = $('grip'); g.textContent = d.gripper.toUpperCase(); g.className = d.gripper;
  };
}
$('reset').onclick = () => { trail = []; if (ws && ws.readyState === 1) ws.send(JSON.stringify({ cmd: 'reset_ee' })); };
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
            .replace("__VIEW_M__", json.dumps(args.ee_span))
        )
        self.monitors: weakref.WeakSet = weakref.WeakSet()
        # per-step delta anchoring (mirrors the device)
        self._prev_pos: np.ndarray | None = None
        self._prev_rot: Rotation | None = None
        # virtual EE: running integral of the mapped per-step dpos (robot frame),
        # so the dashboard can show translation a fixed-scale view (the hand
        # skeleton view auto-recenters every frame and hides global motion).
        self._ee_pos = np.zeros(3, dtype=np.float64)
        # fps accounting
        self._fps = 0
        self._fps_count = 0
        self._fps_t0 = 0.0

    # ---- command computation (identical math to SO101Quest3) ----
    def compute(self, wrist_pos, wrist_quat, joints) -> dict:
        dpos = np.zeros(3)
        drot = np.zeros(3)
        if self._prev_pos is None:
            self._prev_pos = wrist_pos.copy()
            self._prev_rot = Rotation.from_quat(wrist_quat)
            self._ee_pos[:] = 0.0          # re-anchor the virtual EE origin
        else:
            dpos, drot, rot_now = xr_delta_to_world(
                self._prev_pos, self._prev_rot, wrist_pos, wrist_quat,
                self.args.pos_scale, self.args.rot_scale,
                self.args.max_pos_step, self.args.max_rot_step,
            )
            self._prev_pos = wrist_pos.copy()
            self._prev_rot = rot_now
            self._ee_pos += dpos           # integrate the mapped command
        pd = pinch_distance(joints)
        # Absolute wrist orientation mapped through the SAME matrix the command
        # uses, so the dashboard triad shows the wrist's local X/Y/Z as the robot
        # would see them. Columns of ee_basis are those three axes (robot frame).
        ee_basis = _R_XR_TO_ISAAC @ Rotation.from_quat(wrist_quat).as_matrix()
        return {
            "joints": joints.tolist(),
            "wrist_pos": wrist_pos.tolist(),
            "dpos": dpos.tolist(),
            "drot": drot.tolist(),
            "ee_pos": self._ee_pos.tolist(),
            "ee_basis": ee_basis.tolist(),
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
                if cmd.get("cmd") == "reset_ee":
                    self._ee_pos[:] = 0.0
        finally:
            print("[Monitor] Dashboard disconnected")
        return ws

    async def quest_ws(self, request: web.Request) -> web.WebSocketResponse:
        import time

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("[Monitor] Quest connected")
        # fresh anchor for this session
        self._prev_pos = None
        self._prev_rot = None
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
    p.add_argument("--max-pos-step", type=float, default=0.02, help="metres / step clamp")
    p.add_argument("--max-rot-step", type=float, default=0.10, help="rad / step clamp")
    p.add_argument("--ee-span", type=float, default=0.4,
                   help="metres shown across the robot-frame EE views (fixed scale)")
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
