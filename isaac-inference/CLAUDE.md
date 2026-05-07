# Isaac Inference

Autonomous robot policy evaluation and dataset recording for the MasterProject pick-and-place task, using Isaac Sim via LeIsaac and the LeRobot framework.

## Overview

The main script runs a SmolVLA policy in an Isaac Sim environment to autonomously pick oranges and place them in a plate. Successful subtask demonstrations are recorded and uploaded to HuggingFace as a LeRobot dataset.

## Key Files

| File | Purpose |
|------|---------|
| `inference_autonomous_orders.py` | Main entry point — evaluation loop with autonomous subtask ordering |
| `dataset_recorder.py` | Buffers frames per subtask phase, flushes on success, discards on failure |
| `eval_utils.py` | `SubtaskTracker`, `EvaluationTracker`, `HomeChecker`, position helpers |
| `robot_utils.py` | Joint space conversions between LeIsaac and LeRobot conventions |

## Running

```bash
python inference_autonomous_orders.py
```

Press `r` + Enter during a run to force-reset the current episode.

## Configuration (top of `inference_autonomous_orders.py`)

```python
model_id       = "MasterProject2026/Gal-pick-orange-tailedCH20"
n_episodes     = 1000
RECORD_ENABLED = True
RECORD_REPO_ID = "MasterProject2026/Gal-auto-subtasks"
RECORD_LOCAL_PATH = "/home/gal/Documents/MasterProject/synthetic_datasets/recorded_dataset"
```

## Known Issues

- **Kernel soft lockup on ext4 large-file eviction**: LeRobot video encoding writes large intermediates; if these land on ext4 under GPU memory pressure, a kernel bug (6.8.0-110) can pin a CPU core for 100+ seconds and freeze the system. Mitigated by redirecting `TMPDIR` to `/dev/shm` (tmpfs) and calling `torch.cuda.empty_cache()` between episodes.
- **Disk space**: The root partition runs close to capacity (~93% full). Monitor with `df -h /` before long runs.
