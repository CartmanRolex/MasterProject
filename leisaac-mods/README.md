# Quest 3 teleop — quick commands

Full details/tuning notes: see `CLAUDE.md` ("Quest 3 teleoperation" / "Quest 3
calibration monitor" sections). This file is just the commands.

## 1. Run teleop (here, on the desktop, with Isaac Sim)

```bash
# terminal 1 — expose the WebXR server (port must match --quest_port, default 8080)
ngrok http 8080

# terminal 2 — launch teleop + recording
isaac-inference/remote.sh \
  ~/Documents/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-SO101-PickOrange-v0 --teleop_device=quest3 \
  --num_envs=1 --device=cuda --enable_cameras --record \
  --dataset_file=isaac-inference/teleop-datasets/quest3_test.hdf5
```

On the Quest: open the `https://....ngrok-free.app` URL in the Meta Browser,
press the blue button to start hand tracking. On the desktop keyboard: `B`
start control, `R` reset-fail, `N` reset-success. Pinch thumb+index to close
the gripper.

Useful flags: `--quest_port <int>` (default 8080), `--pinch_threshold <m>`
(default 0.035).

## 2. Run the hand-joint visualizer (locally, on Windows, no Isaac/GPU)

```powershell
pip install aiohttp numpy scipy
python quest3_hand_monitor.py
```

```powershell
# second terminal, on the same PC
ngrok http 8080
```

On the Quest: open the ngrok URL, blue button to start tracking.
On the same PC: open `http://localhost:8080/monitor` (local, no ngrok).

The dashboard shows one wrist-centered 3D view (drag to orbit) of the hand
skeleton plus a FWD / RIGHT / UP triad on the wrist, and reports position
(fwd / right / up, m), rotation since origin (deg) and the raw wrist quaternion
from an origin anchored on connect — "Reset origin" re-anchors. If a triad arrow
does not match the physical hand direction, the axis mapping is wrong.

Useful flags: `--port`, `--send-hz`, `--pos-scale`, `--rot-scale`,
`--pinch-threshold` (see `--help`).

## Where to edit if an axis is wrong

- `quest3_webxr.py` → `_R_XR_TO_ISAAC` — shared default, affects the monitor.
- `so101_quest3.py` → `_R_XR_TO_ISAAC_SO101` — device's own axis map, independent
  of the monitor's. Edit this one if the robot moves wrong but the monitor
  looked correct.
