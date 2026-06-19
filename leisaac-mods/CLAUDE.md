# LeIsaac Modifications

Custom patches and modules that extend the upstream [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) library. We cannot push directly to upstream, so all changes to `~/Documents/leisaac/` are mirrored here and replayed on other machines.

## Files

| File | Purpose |
|------|---------|
| `leisaac.patch` | Full diff of our `~/Documents/leisaac/` working tree against the pinned upstream commit. Apply with `git apply` from the leisaac root. |
| `ik_hold_action.py` | Custom IK action: caches joint targets and reapplies them on zero-delta commands, preventing gravity sag when holding a grasped object. |
| `so101_gamepad_v2.py` | Gamepad controller v2 (superseded — kept for reproducibility of older recordings). |
| `so101_gamepad_v3.py` | Gamepad controller v3 — current version; adds roll-lock toggle (press X). |
| `so101_quest3.py` | Meta Quest 3 hand-tracking teleop device. Reuses Gregorio's WebXR server (aiohttp page streamed to the Quest's Meta Browser) and emits a 7D root-frame delta `[dpos(3), drotvec(3), d_gripper]` to a DLS-IK-over-5-joints action; thumb–index pinch closes the gripper. Imports its WebXR page / state / delta math from `quest3_webxr.py`, but defines its **own** axis map `_R_XR_TO_ISAAC_SO101` (identity: XR x=right→idx0, y=up→idx1, z=back→idx2, per the gamepad convention) which it passes to `xr_delta_to_world` as `R`, so the device's hand→arm directions are tuned independently of the monitor. Lives at `devices/so101_quest3.py` (not under `gamepad/`). |
| `quest3_webxr.py` | Shared, Isaac-free WebXR layer for Quest 3 teleop: the WebXR page (`_HTML_TEMPLATE`), the default `_R_XR_TO_ISAAC` frame matrix, `_TeleopState`, the pinch indices, and the `xr_delta_to_world` per-step delta math. `xr_delta_to_world` takes an optional `R` axis map (default `_R_XR_TO_ISAAC`); the monitor uses the default while `so101_quest3.py` passes its own `_R_XR_TO_ISAAC_SO101`, so the device and monitor axis maps are currently independent (to be reconciled when the monitor is revisited). Imported by **both** `so101_quest3.py` (in the leisaac tree) and `quest3_hand_monitor.py`. Lives at `devices/quest3_webxr.py`. |
| `quest3_hand_monitor.py` | Standalone Quest 3 calibration tool (no Isaac, no GPU). Runs the same WebXR server and serves a self-contained browser dashboard (`/monitor`) that live-plots the right-hand skeleton plus the exact command the device would emit (dpos/drot, pinch distance, gripper state). Used to calibrate headset placement and the XR→Isaac mapping without launching the sim. MasterProject-only — not deployed into the leisaac tree. |

## What the patch contains

The patch combines several layers of changes on top of upstream leisaac:

- **Privileged observation frames** (required by `inference_autonomous_orders.py` and `inference_flat_prompt.py`): adds `gripper_tip` and `jaw_tip` `FrameTransformerCfg` entries to `SingleArmTaskSceneCfg.ee_frame`. Without these, `eval_utils.py:563` crashes with `'gripper_tip' is not in list`.
- **Contact sensors**: `gripper_contact` and `jaw_contact` `ContactSensorCfg` filtered to the three orange prim paths, plus `activate_contact_sensors=True` on `SO101_FOLLOWER_CFG`, so we can read gripper/jaw–orange contact forces during inference.
- **Camera randomization baseline**: caches the initial camera USD pose inside `randomize_camera_uniform` so reset-time front-camera randomization stays centered on the original pose instead of compounding into a random walk on machines where Isaac refreshes `asset.data.pos_w`.
- **Gamepad v2 / v3 and Quest 3 wiring**: registers the teleop device names (`gamepad_v2`, `gamepad_v3`, `quest3`) in `scripts/environments/teleoperation/teleop_se3_agent.py`, `source/leisaac/leisaac/devices/__init__.py`, `source/leisaac/leisaac/devices/gamepad/__init__.py`, the `init_action_cfg` / `preprocess_device_action` branches in `devices/action_process.py`, the `use_teleop_device` list in `tasks/template/single_arm_env_cfg.py`, and the gravity/effort-limit branches in `utils/env_utils.py`. `quest3` also adds `--quest_port` / `--pinch_threshold` CLI args and a 7D action branch.
- **Misc**: dataset conversion tweaks in `scripts/convert/isaaclab2lerobotv3.py`, action-process and robot-utils updates.

## Pinned upstream commit

The patch is generated against upstream commit **`241779b`** (`add CONTRIBUTING.md. (#147)`). Newer upstream commits introduce conflicts in `source/leisaac/leisaac/devices/__init__.py` — pin the leisaac checkout to `241779b` before applying.

## Installation on a new machine

```bash
# 1. Clone (or fetch) leisaac and pin to the supported upstream commit
cd ~/Documents
git clone https://github.com/LightwheelAI/leisaac.git   # skip if already cloned
cd leisaac
git fetch origin
git checkout 241779b

# 2. Apply the tracked-file patch
git apply ~/Documents/MasterProject/leisaac-mods/leisaac.patch

# 3. Drop in the untracked Python modules
cp ~/Documents/MasterProject/leisaac-mods/ik_hold_action.py    source/leisaac/leisaac/devices/
cp ~/Documents/MasterProject/leisaac-mods/so101_gamepad_v2.py  source/leisaac/leisaac/devices/gamepad/
cp ~/Documents/MasterProject/leisaac-mods/so101_gamepad_v3.py  source/leisaac/leisaac/devices/gamepad/
cp ~/Documents/MasterProject/leisaac-mods/so101_quest3.py      source/leisaac/leisaac/devices/
cp ~/Documents/MasterProject/leisaac-mods/quest3_webxr.py      source/leisaac/leisaac/devices/
```

(`quest3_hand_monitor.py` is a MasterProject-only calibration tool — it is **not**
copied into the leisaac tree; run it in place from `leisaac-mods/`.)

After step 2, `git status` inside `~/Documents/leisaac/` should show the same modified files listed in the patch header. After step 3, `inference_autonomous_orders.py` should no longer crash on `frame_names.index("gripper_tip")`.

## Quest 3 teleoperation

The `quest3` device streams Meta Quest 3 hand tracking to the SO-101 over WebXR.
WebXR `immersive-ar` requires an **HTTPS** secure context, so a plain LAN URL
will not work — expose the server with ngrok:

```bash
# terminal 1 — expose the WebXR server (port matches --quest_port, default 8080)
ngrok http 8080
# terminal 2 — launch teleop + recording
isaac-inference/remote.sh \
  ~/Documents/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-SO101-PickOrange-v0 --teleop_device=quest3 \
  --num_envs=1 --device=cuda --enable_cameras --record \
  --dataset_file=isaac-inference/teleop-datasets/quest3_test.hdf5
```

Open the ngrok `https://….ngrok-free.app` URL in the Quest's Meta Browser, press
the blue button to start hand tracking, then use the desktop keyboard B/R/N to
start control / reset-fail / reset-success exactly as with the gamepad. Pinch
thumb+index to close the gripper. Requires `ngrok`, `aiohttp`, and `scipy`. The
hand→arm mapping (`pos_scale`, `rot_scale`, `pinch_threshold_m`, per-step clamps,
and the `_R_XR_TO_ISAAC_SO101` axis map) is tuned in `so101_quest3.py`. The axis
map follows the `so101_gamepad_v3` index convention — position idx0=right,
idx1=up, idx2=back (forward=−idx2). If a motion axis comes out mirrored/swapped
on the robot, flip the corresponding row/sign of `_R_XR_TO_ISAAC_SO101`; if it
looks right in the monitor but wrong on the robot, the suspect is the
world→root step (`_world_to_root`) / base orientation, not the matrix.

## Quest 3 calibration monitor

`quest3_hand_monitor.py` is a standalone tool (no Isaac, no GPU) for calibrating
headset placement and the hand→arm mapping **without launching the sim** — useful
for Option B, where the headset sits on a stand tracking your hands while you view
the robot with your own eyes. It runs the same WebXR server the device uses and
serves a self-contained browser dashboard. It is cross-platform (Linux/Windows).

It needs only `aiohttp`, `numpy`, `scipy` — no Isaac, no GPU — so the recommended
way is to **run it on the same PC as the Quest** (e.g. your Windows laptop). Then
the dashboard is local (`localhost`) and the only thing that crosses the network
is the Quest's hand stream. The leisaac desktop/server is not involved at all.

### Recommended: run on your own PC, expose with ngrok (no USB)

WebXR `immersive-ar` requires an HTTPS secure context, and the Quest is on a
different network path than the PC, so the Quest reaches the local server through
an ngrok HTTPS tunnel. The dashboard does **not** go through ngrok — you open it
directly on the same PC at `localhost`.

```bash
# On the PC physically with you (Windows: use PowerShell / a terminal):
# 1. get the two files (clone the repo, or `git pull` if already cloned)
git clone git@github.com:CartmanRolex/MasterProject.git
cd MasterProject/leisaac-mods
# 2. install the three deps into any Python 3.10+ (a venv is fine)
pip install aiohttp numpy scipy
# 3. start the monitor (binds 0.0.0.0:8080 on THIS PC)
python quest3_hand_monitor.py
# 4. in a second terminal, expose port 8080 over HTTPS
ngrok http 8080            # one-time setup: `ngrok config add-authtoken <token>`
```

Then:
- **On the Quest:** open the `https://….ngrok-free.app` URL ngrok prints, in the
  Meta Browser, and press the blue button to start hand tracking.
- **On your PC:** open `http://localhost:8080/monitor` in any browser.

So the Quest reaches your PC via `Quest → ngrok cloud → your-PC:8080/ws`, while the
dashboard is just `your-PC:8080/monitor` opened locally. ngrok is only the Quest's
way in; nothing else is tunneled. (A USB alternative — `adb reverse tcp:8080
tcp:8080` then `http://localhost:8080` in the Quest browser — avoids the ngrok
cloud hop for slightly lower latency, but needs a cable and developer mode.)

### Alternative: run on the leisaac desktop over SSH

If you instead run it on the desktop, the Quest reaches it via `ngrok http 8080`
(run on the desktop) and you view the dashboard through an SSH tunnel:
`ssh -L 8080:localhost:8080 <user>@<desktop>`, then `http://localhost:8080/monitor`
locally. The SSH `-L` tunnel carries the dashboard + its WebSocket fine (all TCP).
On the leisaac env the deps are already present:
`conda run -n leisaac_envhub python quest3_hand_monitor.py`.

### What the dashboard shows / how it transfers

Two **fixed-scale robot-frame views** (TOP = forward×lateral, SIDE = forward×up)
plot a **virtual end-effector**: the yellow dot is the running integral of the
mapped per-step `dpos`, a green arrow shows the current step's `dpos`, and an
`x`/`y`/`z` triad shows the wrist's orientation mapped through `_R_XR_TO_ISAAC`
into the robot frame. A third panel keeps one auto-fit hand-skeleton view (the
hand view deliberately re-centers every frame, so it hides translation — that is
why the EE views exist). The side panel lists `dpos`/`drot`, the integrated `ee`,
pinch vs threshold, and an OPEN/CLOSED gripper. A **Reset EE origin** button
re-zeros the integrator (also sent automatically when tracking re-anchors).

The axes are labelled in the **gamepad convention** so they match the working
`so101_gamepad_v3` teleop: position index **0 = up**, **1 = lateral**,
**2 = back** (forward = −idx2); rotation = [roll, pitch, yaw]. Move your hand and
read which robot axis actually responds — that exposes a wrong/mirrored/swapped
axis, which you then fix by editing the rows/signs of `_R_XR_TO_ISAAC` in
`quest3_webxr.py` (shared, so `so101_quest3.py` inherits the fix on restart). If
the EE view looks correct but the real robot moves wrong, the bug is downstream
in `so101_quest3.py`'s `_world_to_root` / base orientation, not the matrix.

The command is computed with the same `xr_delta_to_world` / `pinch_distance` as
the device. CLI flags (`--pos-scale`, `--rot-scale`, `--pinch-threshold`,
`--max-pos-step`, `--max-rot-step`, `--ee-span` [metres across the EE views],
`--port`, `--send-hz`) default to the device's values.

## Regenerating the patch (Desktop side, when leisaac is edited)

When you make further changes inside `~/Documents/leisaac/`, refresh this directory so other machines can pull them:

```bash
cd ~/Documents/leisaac
git diff origin/main > ~/Documents/MasterProject/leisaac-mods/leisaac.patch

# also refresh the untracked module copies if they changed
cp source/leisaac/leisaac/devices/ik_hold_action.py            ~/Documents/MasterProject/leisaac-mods/
cp source/leisaac/leisaac/devices/gamepad/so101_gamepad_v2.py  ~/Documents/MasterProject/leisaac-mods/
cp source/leisaac/leisaac/devices/gamepad/so101_gamepad_v3.py  ~/Documents/MasterProject/leisaac-mods/
cp source/leisaac/leisaac/devices/so101_quest3.py              ~/Documents/MasterProject/leisaac-mods/
cp source/leisaac/leisaac/devices/quest3_webxr.py             ~/Documents/MasterProject/leisaac-mods/
```

Then commit & push from `MasterProject/`.

## Keeping this file current

Update this file **and** `AGENTS.md` in the same commit as any structural change: new patch content, new module copies, changed pinned upstream commit. The `.md` and the code must always agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**
