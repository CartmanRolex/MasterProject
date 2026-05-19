# MasterProject — Project Context

## What this project is
SmolVLA fine-tuned for pick-and-place on an SO-101 arm in Isaac Sim.
Task: pick 3 oranges, place in plate. VLA handles manipulation subtasks;
an algorithmic orchestrator sequences them using privileged Isaac state.

## Core conceptual framing (READ BEFORE EDITING ORCHESTRATOR OR REPORT)

The system has THREE distinct mechanisms that were previously all called
"recovery." Do not conflate them. Code, comments, and report text must
use these names:

1. **Local retry** — true recovery. Orange slips during LIFT/PLACE →
   return to GRASP for the SAME orange. Original goal preserved.

2. **Spatial reset** — a PRECONDITION, not recovery. VLA language
   conditioning only switches targets reliably when the arm is far from
   all oranges. Implemented as a SCRIPTED joint-space interpolation to
   home (NOT a VLA prompt). Required before any target change.

3. **Target redirection** — goal ABANDONMENT, not recovery. Repeated
   GRASP failure on orange A → give up on A, attempt orange B. Improves
   task completion rate; original sub-goal is dropped.

Invariant: any change of language target must be preceded by a spatial
reset. Same target after a slip does not need one.

The VLA is responsible for: GRASP(target), LIFT, PLACE.
The orchestrator is responsible for: subtask sequencing, outcome
classification, spatial resets, and target selection.

## Terminology — do not drift
- "Recovery" refers ONLY to local retry. Not to spatial reset, not to
  redirection.
- "Subtask" = one of GRASP / LIFT / PLACE. HOME is a scripted primitive,
  not a subtask.
- "Phantom grasp" = gripper closed on air; detected via gripper force.

## Key finding to preserve in writing
VLA language conditioning is spatially context-dependent. Target
switching fails when the gripper is near a non-target orange. This
motivates the scripted spatial reset.

---

# Isaac Inference (desktop)

Autonomous robot policy evaluation and dataset recording for the
MasterProject pick-and-place task, using Isaac Sim via LeIsaac and
the LeRobot framework.

## Current priorities (May 2026)

In order. Do not start item N+1 until N is done and acknowledged.

1. **Refactor `inference_autonomous_orders.py` around the three
   mechanisms.** Replace ad-hoc phase transitions with an explicit
   state machine. After each subtask, classify outcome and return one
   of: RETRY_SAME, REDIRECT (with new target), or ABANDON_ORANGE.
   GO_HOME is a scripted primitive inserted before any target change.

2. **Pull HOME out of the VLA entirely.** Replace the
   `"Go back to start position"` prompt with a scripted joint-space
   interpolation. The VLA never executes return-to-home.

3. **Run the full 100-episode eval** on the refactored orchestrator
   with `Gal-pick-orange-tailedCH20`. Confirm the GRASP-with-1-placed
   bottleneck from the partial 3-run result.

4. **Phase 5 (autonomous data generation)** — do not start until 1–3
   are clean. The auto-collected dataset must be a clean reflection
   of the refactored orchestrator's behavior.

## Out of scope right now
- Dataset editor changes (`dataset-editor/`)
- New teleop work (`leisaac-mods/`)
- Switching the orchestrator to a VLM (privileged state stays)
- Hyperparameter sweeps on training

## Key files (current work)

| File | Purpose |
|------|---------|
| `inference_autonomous_orders.py` | **Main entry point** — orchestrator + eval loop. Target architecture: state machine over GRASP / LIFT / PLACE subtasks, with scripted GO_HOME primitive and explicit RETRY_SAME / REDIRECT / ABANDON_ORANGE transitions. |
| `dataset_recorder.py` | Buffers frames per subtask; commits on success, discards on failure. Crash-safe via `checkpoint.json`. |
| `eval_utils.py` | `SubtaskTracker`, `EvaluationTracker`, `HomeChecker`, position/scene helpers. |
| `robot_utils.py` | Joint-space conversions: LeIsaac (radians) ↔ LeRobot (normalized degrees). |
| `remote.sh` | Isaac Sim launcher — sets `ENABLE_LIVESTREAM`, `LEISAAC_ASSETS_ROOT`, then calls `python "$@"`. |
| `commands.txt` | Reference commands (training, inference, teleop, dataset conversion). |

## Legacy / alternative entry points (do not modify without reason)

Moved to `legacy/` to keep root clean.

| File | Purpose |
|------|---------|
| `legacy/inference_prompts.py` | SmolVLA with 3 fixed prompts cycling through phases. |
| `legacy/inference_smolvla.py` | Baseline — single SmolVLA policy, fixed instruction, no subtasks. |
| `legacy/inference_act.py` | ACT policy evaluation. |
| `legacy/policy_inference.py` | Legacy server-mode via `lerobot.async_inference.policy_server`. |

## Running

```bash
./remote.sh inference_autonomous_orders.py
```

Press `r` + Enter during a run to force-reset the current episode.

## Configuration (top of `inference_autonomous_orders.py`)

```python
model_id          = "MasterProject2026/Gal-pick-orange-tailedCH20"
n_episodes        = 1000
RECORD_ENABLED    = True
RECORD_REPO_ID    = "MasterProject2026/Gal-auto-subtasks"
RECORD_LOCAL_PATH = "/home/gal/Documents/MasterProject/isaac-inference/synthetic_datasets/recorded_dataset"
```

## Eval results

All evaluation summaries land in `results/` next to this file,
regardless of working directory (`eval_utils.py` uses
`Path(__file__).parent / "results"`). Results are git-tracked.
`results/plot.py` generates comparison bar charts across models.

## Gitignored paths
- `${data}/` — Isaac Sim NvStreamer `.etli` streaming logs (auto-generated)
- `teleop-datasets/` — 236 GB HDF5 teleoperation datasets
- `synthetic_datasets/` — local LeRobot-format recorded dataset
- `__pycache__/`

## Known issues
- **Kernel soft lockup on ext4 large-file eviction**: LeRobot video
  encoding writes large intermediates to disk; under GPU memory
  pressure this can pin a CPU core for 100+ seconds and freeze the
  system. Mitigated: `TMPDIR` redirected to `/dev/shm` (tmpfs) and
  `torch.cuda.empty_cache()` called between episodes.
- **Disk space**: Root partition was ~93% full. Monitor with
  `df -h /` before long runs.