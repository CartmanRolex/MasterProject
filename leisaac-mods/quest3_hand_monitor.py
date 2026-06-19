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
         display: flex; flex-direction: column; height: 100vh; padding: 12px; gap: 12px; }
  h1 { font-size: 18px; font-weight: 700; }
  #wrap { display: flex; gap: 12px; flex: 1; min-height: 0; }
  .views { display: flex; gap: 12px; flex: 2; }
  .panel { background: #16162a; border-radius: 10px; padding: 8px; display: flex;
           flex-direction: column; gap: 6px; flex: 1; min-width: 0; }
  .panel h2 { font-size: 12px; opacity: 0.6; font-weight: 600; }
  canvas { background: #0a0a14; border-radius: 6px; width: 100%; height: 100%;
           display: block; flex: 1; min-height: 0; }
  #side { flex: 1; display: flex; flex-direction: column; gap: 10px; min-width: 240px; }
  .stat { background: #16162a; border-radius: 10px; padding: 10px 12px; }
  .stat .lbl { font-size: 11px; opacity: 0.55; text-transform: uppercase; letter-spacing: 0.5px; }
  .row { display: flex; justify-content: space-between; font-size: 14px; margin-top: 4px; }
  .row b { color: #9fa8da; }
  #grip { font-size: 26px; font-weight: 800; text-align: center; padding: 14px;
          border-radius: 10px; transition: background 0.05s; }
  .open   { background: #1b3a1b; color: #7ee787; }
  .closed { background: #4a1414; color: #ff8b8b; }
  #conn { font-size: 12px; opacity: 0.6; }
  .x { color: #ff6b6b; } .y { color: #6bff9e; } .z { color: #6bb8ff; }
</style>
</head>
<body>
  <h1>Quest3 Hand Monitor <span id="conn">connecting…</span></h1>
  <div id="wrap">
    <div class="views">
      <div class="panel"><h2>TOP view — X right, Z (depth) up</h2><canvas id="cTop"></canvas></div>
      <div class="panel"><h2>FRONT view — X right, Y up</h2><canvas id="cFront"></canvas></div>
    </div>
    <div id="side">
      <div id="grip" class="open">GRIPPER</div>
      <div class="stat">
        <div class="lbl">Pinch (thumb–index)</div>
        <div class="row"><span>distance</span><b id="pd">– m</b></div>
        <div class="row"><span>threshold</span><b id="pt">– m</b></div>
      </div>
      <div class="stat">
        <div class="lbl">dpos — Isaac world, per step</div>
        <div class="row"><span class="x">dx (fwd)</span><b id="dx">–</b></div>
        <div class="row"><span class="y">dy (left)</span><b id="dy">–</b></div>
        <div class="row"><span class="z">dz (up)</span><b id="dz">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">drot — rotvec, per step</div>
        <div class="row"><span class="x">rx</span><b id="rx">–</b></div>
        <div class="row"><span class="y">ry</span><b id="ry">–</b></div>
        <div class="row"><span class="z">rz</span><b id="rz">–</b></div>
      </div>
      <div class="stat">
        <div class="lbl">stream</div>
        <div class="row"><span>fps</span><b id="fps">–</b></div>
        <div class="row"><span>wrist pos (xr)</span><b id="wp">–</b></div>
      </div>
    </div>
  </div>
<script>
const BONES = __BONES__;
const $ = id => document.getElementById(id);
const conn = $('conn');

function fitCanvas(c) {
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, r.width | 0); c.height = Math.max(1, r.height | 0);
}
const cTop = $('cTop'), cFront = $('cFront');
const gTop = cTop.getContext('2d'), gFront = cFront.getContext('2d');
window.addEventListener('resize', () => { fitCanvas(cTop); fitCanvas(cFront); });
fitCanvas(cTop); fitCanvas(cFront);

// Draw a 25-joint hand into ctx using accessor (j)->[h,v] in metres.
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
  const Y = hv => pad + h / 2 - (hv[1] - cV) * sc;   // flip: screen y down
  // bones
  ctx.strokeStyle = '#5c6bc0'; ctx.lineWidth = 2;
  for (const [a, b] of BONES) {
    ctx.beginPath(); ctx.moveTo(X(pts[a]), Y(pts[a])); ctx.lineTo(X(pts[b]), Y(pts[b])); ctx.stroke();
  }
  // joints
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
    const d = JSON.parse(ev.data);
    const j = d.joints;
    drawHand(gTop,   cTop,   j, p => [p[0], p[2]]);   // X, Z
    drawHand(gFront, cFront, j, p => [p[0], p[1]]);   // X, Y
    const f = (v, n=4) => (v >= 0 ? ' ' : '') + v.toFixed(n);
    $('pd').textContent = f(d.pinch_d, 4) + ' m';
    $('pt').textContent = f(d.pinch_threshold, 4) + ' m';
    $('dx').textContent = f(d.dpos[0]); $('dy').textContent = f(d.dpos[1]); $('dz').textContent = f(d.dpos[2]);
    $('rx').textContent = f(d.drot[0]); $('ry').textContent = f(d.drot[1]); $('rz').textContent = f(d.drot[2]);
    $('fps').textContent = d.fps;
    $('wp').textContent = d.wrist_pos.map(v => v.toFixed(2)).join(', ');
    const g = $('grip');
    g.textContent = d.gripper.toUpperCase();
    g.className = d.gripper;
  };
}
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
        self.dashboard_html = _DASHBOARD_HTML.replace("__BONES__", json.dumps(_BONES))
        self.monitors: weakref.WeakSet = weakref.WeakSet()
        # per-step delta anchoring (mirrors the device)
        self._prev_pos: np.ndarray | None = None
        self._prev_rot: Rotation | None = None
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
        else:
            dpos, drot, rot_now = xr_delta_to_world(
                self._prev_pos, self._prev_rot, wrist_pos, wrist_quat,
                self.args.pos_scale, self.args.rot_scale,
                self.args.max_pos_step, self.args.max_rot_step,
            )
            self._prev_pos = wrist_pos.copy()
            self._prev_rot = rot_now
        pd = pinch_distance(joints)
        return {
            "joints": joints.tolist(),
            "wrist_pos": wrist_pos.tolist(),
            "dpos": dpos.tolist(),
            "drot": drot.tolist(),
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
            async for _ in ws:  # dashboard is receive-only; just keep the socket open
                pass
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
