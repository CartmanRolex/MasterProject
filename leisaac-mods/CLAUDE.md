# LeIsaac Modifications

Custom patches and modules that extend the LeIsaac library. Apply these to the local `leisaac` checkout at `~/Documents/leisaac/`.

## Files

| File | Purpose |
|------|---------|
| `leisaac.patch` | Git patch for LeIsaac — apply with `git apply` from the leisaac root |
| `ik_hold_action.py` | Custom IK action: caches joint targets and reapplies them on zero-delta commands, preventing gravity sag when holding a grasped object |
| `so101_gamepad_v2.py` | Gamepad controller v2 (superseded) |
| `so101_gamepad_v3.py` | Gamepad controller v3 — current version; adds roll-lock toggle (press X) |

## Applying the Patch

```bash
cd ~/Documents/leisaac
git apply ~/Documents/MasterProject/leisaac-mods/leisaac.patch
```

## Installing Custom Modules

Copy into the LeIsaac source tree before running Isaac Sim:

```bash
cp ik_hold_action.py ~/Documents/leisaac/source/leisaac/envs/mdp/actions/
cp so101_gamepad_v3.py ~/Documents/leisaac/source/leisaac/devices/gamepad/
```

Refer to `leisaac.patch` for the exact import paths expected by each module.
