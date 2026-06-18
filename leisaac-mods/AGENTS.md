# LeIsaac Modifications

Custom patches and modules that extend the upstream [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) library. We cannot push directly to upstream, so all changes to `~/Documents/leisaac/` are mirrored here and replayed on other machines.

## Files

| File | Purpose |
|------|---------|
| `leisaac.patch` | Full diff of our `~/Documents/leisaac/` working tree against the pinned upstream commit. Apply with `git apply` from the leisaac root. |
| `ik_hold_action.py` | Custom IK action: caches joint targets and reapplies them on zero-delta commands, preventing gravity sag when holding a grasped object. |
| `so101_gamepad_v2.py` | Gamepad controller v2 (superseded — kept for reproducibility of older recordings). |
| `so101_gamepad_v3.py` | Gamepad controller v3 — current version; adds roll-lock toggle (press X). |
| `so101_quest3.py` | Meta Quest 3 hand-tracking teleop device. Reuses Gregorio's WebXR server (aiohttp page streamed to the Quest's Meta Browser) and emits a 7D root-frame delta `[dpos(3), drotvec(3), d_gripper]` to a DLS-IK-over-5-joints action; thumb–index pinch closes the gripper. Lives at `devices/so101_quest3.py` (not under `gamepad/`). |

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
```

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
and the `_R_XR_TO_ISAAC` axis signs) is tuned live in `so101_quest3.py`.

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
```

Then commit & push from `MasterProject/`.

## Keeping this file current

Update this file **and** `CLAUDE.md` in the same commit as any structural change: new patch content, new module copies, changed pinned upstream commit. The `.md` and the code must always agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**
