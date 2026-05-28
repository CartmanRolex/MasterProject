# LeIsaac Modifications

Custom patches and modules that extend the upstream [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) library. We cannot push directly to upstream, so all changes to `~/Documents/leisaac/` are mirrored here and replayed on other machines.

## Files

| File | Purpose |
|------|---------|
| `leisaac.patch` | Full diff of our `~/Documents/leisaac/` working tree against the pinned upstream commit. Apply with `git apply` from the leisaac root. |
| `ik_hold_action.py` | Custom IK action: caches joint targets and reapplies them on zero-delta commands, preventing gravity sag when holding a grasped object. |
| `so101_gamepad_v2.py` | Gamepad controller v2 (superseded — kept for reproducibility of older recordings). |
| `so101_gamepad_v3.py` | Gamepad controller v3 — current version; adds roll-lock toggle (press X). |

## What the patch contains

The patch combines several layers of changes on top of upstream leisaac:

- **Privileged observation frames** (required by `inference_autonomous_orders.py` and `inference_flat_prompt.py`): adds `gripper_tip` and `jaw_tip` `FrameTransformerCfg` entries to `SingleArmTaskSceneCfg.ee_frame`. Without these, `eval_utils.py:563` crashes with `'gripper_tip' is not in list`.
- **Contact sensors**: `gripper_contact` and `jaw_contact` `ContactSensorCfg` filtered to the three orange prim paths, plus `activate_contact_sensors=True` on `SO101_FOLLOWER_CFG`, so we can read gripper/jaw–orange contact forces during inference.
- **Gamepad v2 / v3 wiring**: registers the new teleop device names in `scripts/environments/teleoperation/teleop_se3_agent.py`, `source/leisaac/leisaac/devices/__init__.py`, `source/leisaac/leisaac/devices/gamepad/__init__.py`, and the gravity/effort-limit branches in `utils/env_utils.py`.
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
```

After step 2, `git status` inside `~/Documents/leisaac/` should show the same modified files listed in the patch header. After step 3, `inference_autonomous_orders.py` should no longer crash on `frame_names.index("gripper_tip")`.

## Regenerating the patch (Desktop side, when leisaac is edited)

When you make further changes inside `~/Documents/leisaac/`, refresh this directory so other machines can pull them:

```bash
cd ~/Documents/leisaac
git diff origin/main > ~/Documents/MasterProject/leisaac-mods/leisaac.patch

# also refresh the untracked module copies if they changed
cp source/leisaac/leisaac/devices/ik_hold_action.py            ~/Documents/MasterProject/leisaac-mods/
cp source/leisaac/leisaac/devices/gamepad/so101_gamepad_v2.py  ~/Documents/MasterProject/leisaac-mods/
cp source/leisaac/leisaac/devices/gamepad/so101_gamepad_v3.py  ~/Documents/MasterProject/leisaac-mods/
```

Then commit & push from `MasterProject/`.
