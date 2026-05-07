# Isaac Inference

Autonomous robot policy evaluation and dataset recording for the MasterProject pick-and-place task, using Isaac Sim via LeIsaac and the LeRobot framework.

## Overview

Scripts in this directory run policy evaluation loops inside Isaac Sim. The main script (`inference_autonomous_orders.py`) uses a SmolVLA policy with autonomous subtask sequencing (GRASP → LIFT → PLACE → HOME). Successful subtask demonstrations are recorded and uploaded to HuggingFace as a LeRobot dataset.

## Key Files

| File | Purpose |
|------|---------|
| `inference_autonomous_orders.py` | **Main entry point** — evaluation loop with autonomous subtask ordering (GRASP / LIFT / PLACE / HOME / RECOVERY phases) |
| `inference_prompts.py` | Alternative mode — SmolVLA with 3 fixed language prompts cycling through subtask phases |
| `inference_smolvla.py` | Simple baseline — single SmolVLA policy with a fixed instruction, no subtask structure |
| `inference_act.py` | ACT policy evaluation |
| `policy_inference.py` | Legacy server-mode approach — policy runs in a separate process via `lerobot.async_inference.policy_server` |
| `dataset_recorder.py` | Buffers frames per subtask phase; flushes to LeRobot dataset on success, discards on failure |
| `eval_utils.py` | `SubtaskTracker`, `EvaluationTracker`, `HomeChecker`, position/scene helpers |
| `robot_utils.py` | Joint space conversions between LeIsaac (radians) and LeRobot (normalized degrees) conventions |
| `remote.sh` | Isaac Sim launcher wrapper — sets `ENABLE_LIVESTREAM`, `LEISAAC_ASSETS_ROOT`, then calls `python "$@"` |
| `commands.txt` | Reference commands for training, inference, teleoperation, and dataset conversion |

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

## Eval Results

All evaluation summaries always land in `results/` next to this file, regardless of working directory (`eval_utils.py` uses `Path(__file__).parent / "results"`). Results are git-tracked. `results/plot.py` generates comparison bar charts across models.

## Gitignored Paths

- `${data}/` — Isaac Sim NvStreamer `.etli` streaming logs (auto-generated, deleted and gitignored)
- `teleop-datasets/` — 236 GB HDF5 teleoperation datasets
- `synthetic_datasets/` — local LeRobot-format recorded dataset (output of `dataset_recorder.py`)
- `__pycache__/` — Python bytecode

## Known Issues

- **Kernel soft lockup on ext4 large-file eviction**: LeRobot video encoding writes large intermediates to disk; under GPU memory pressure this can pin a CPU core for 100+ seconds and freeze the system. Mitigated: `TMPDIR` redirected to `/dev/shm` (tmpfs) and `torch.cuda.empty_cache()` called between episodes.
- **Disk space**: Root partition was ~93% full. Monitor with `df -h /` before long runs.
