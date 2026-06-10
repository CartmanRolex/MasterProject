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
   all oranges. Current implementation is fully scripted: first move
   only `shoulder_lift` toward the episode-start value, then move all
   joints to home. Required before any target change.

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

## Status (June 2026)

Implementation and evaluation are complete. The orchestrator refactor, scripted spatial reset, full 100-episode seeded evals, and Phase 5 autonomous data generation are all done. Current focus is the report (`report/` on the laptop).

## Key files (current work)

| File | Purpose |
|------|---------|
| `inference_autonomous_orders.py` | **Main entry point** — orchestrator + eval loop. Target architecture: state machine over GRASP / LIFT / PLACE subtasks, with two-stage scripted GO_HOME reset and explicit RETRY_SAME / REDIRECT / ABANDON_ORANGE transitions. |
| `dataset_recorder.py` | Buffers frames per subtask; commits on success, discards on failure. Crash-safe via `checkpoint.json`. |
| `eval_utils.py` | `SubtaskTracker`, `EvaluationTracker`, `EpisodeStory`, `HomeChecker`, position/scene helpers. |
| `robot_utils.py` | Joint-space conversions: LeIsaac (radians) ↔ LeRobot (normalized degrees). |
| `remote.sh` | Isaac Sim launcher — sets `ENABLE_LIVESTREAM`, `LEISAAC_ASSETS_ROOT`, then calls `python "$@"`. |
| `commands.txt` | Reference commands (training, inference, teleop, dataset conversion). |
| `inference_privileged_grasp.py` | Privileged grasp policy — uses DLS (Damped Least Squares) IK to position the gripper above each orange's exact XY from Isaac privileged state. Fully scripted/geometric; no VLA. |
| `inference_flat_prompt.py` | Flat single-instruction SmolVLA eval — one fixed prompt ("Place the orange into plate"), no subtask sequencing. Used to evaluate the no-lang-no-home model variant. |
| `inference_act_flat_prompt.py` | ACT policy eval — same flat single-instruction structure as `inference_flat_prompt.py` but uses `ACTPolicy` with chunked action execution (`predict_action_chunk`; chunk size comes from `policy.config.chunk_size`). Saves to `results/<model>/act_checkpoint.json` and `act_latest.txt`. |
| `phase_monitor.py` | `PhaseMonitor` — passive observer-inferred subtask trace for flat (no-language) policies, derived from physics state alone (`SubtaskTracker`, orange/plate positions). Used by `inference_flat_prompt.py` and `inference_act_flat_prompt.py` to produce per-episode subtask traces for policies that don't expose intent. |
| `test_phase_monitor.py` | Unit tests for `PhaseMonitor` trace generation. |

## Legacy / alternative entry points (do not modify without reason)

Moved to `legacy/` to keep root clean.

| File | Purpose |
|------|---------|
| `legacy/inference_prompts.py` | SmolVLA with 3 fixed prompts cycling through phases. |
| `legacy/inference_smolvla.py` | Baseline — single SmolVLA policy, fixed instruction, no subtasks. |
| `legacy/inference_act.py` | ACT policy evaluation. |
| `legacy/policy_inference.py` | Legacy server-mode via `lerobot.async_inference.policy_server`. |
| `legacy/inference_explore.py` | Exploratory/experimental inference variant. Kept for reference; not part of any current eval. |

## Data pipeline scripts

One-shot scripts for building and transforming the training dataset. Run from `isaac-inference/` on the desktop. Output goes to `synthetic_datasets/`.

| Script | Purpose |
|--------|---------|
| `merge_datasets.py` | Concatenate two or more LeRobot v3.0 datasets into one. Episode indices are shifted and task strings are unioned. Re-encodes video to h264 (faster decode on non-RTX-40 hardware). Writes `meta/stats.json` (scalar features recomputed from the merged data parquet; image features count-weighted-aggregated from the sources). |
| `balance_dataset.py` | Create a balanced subset across the 9 subtask slots (3 oranges × Grasp / Pick up / Place). N = min episode count per bucket; total output = 9×N. HOME episodes excluded. |
| `consolidate_dataset_videos.py` | Merge per-episode chunk files into one large file per camera. Eliminates torchcodec decoder re-init overhead at training time. Idempotent — safe to re-run. |
| `fix_merged_lengths.py` | Repair per-episode length drift that can appear after `consolidate_dataset_videos.py`. Run if episode lengths in `meta/episodes.parquet` look inconsistent. |
| `tail_split.py` | Properly "tail" a teleop-derived dataset by **appending** `N=20` freeze frames to every episode (inverse of `strip_lang_and_tail.py`). Drops `"Go back to start position"` episodes and reproduces `dataset_recorder.py`'s per-subtask freeze: GRASP / LIFT hold the last commanded gripper action (closing force), PLACE freezes fully (action = state). Clones the last video frame (ffmpeg `tpad`), reconciles clip lengths, consolidates, writes `stats.json`. Used to rebuild `Gal_split` → `Gal_split_tailed` before merging into `Gal-merged-tailed-auto`. |
| `strip_lang_and_tail.py` | Produce a no-language-conditioning clone with the last 20 frozen frames of every episode removed. Hardcoded paths: `Gal-merged-tailed-auto` → `Gal-merged-tailed-auto-no-lang-no-home`. |
| `plot_dataset_stats.py` | Plot task distribution and composition statistics for a synthetic dataset. |

## Maintenance / one-shot utilities

| Script | Purpose |
|--------|---------|
| `recover_episodes.py` | Repair corrupted episode parquet files. Originally written to fix `Gal-auto-subtasks2`. Run once as needed; not part of normal workflow. |
| `debug_camera_drift.py` | Diagnostic: loads pick-orange env and runs 200 rapid resets, logging front camera world position before/after each `randomize_camera_uniform` call. Outputs `camera_drift_log_<hostname>.txt`. Run on both desktop and laptop to compare accumulation behaviour. |
| `overnight_eval_queue.py` | Preflight checks and tmux-based queue for running multiple seeded eval jobs overnight. Launches shards sequentially with stall detection and restart logic. |
| `rerun_gal_split_nolang_correct_prompt.py` | One-shot rerun of the `Gal_split_nolang` flat eval in 3 shards with the corrected prompt ("Place the orange into plate"). Run once to produce the final result for that model. |

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
Seeded benchmark runs use `eval_seeds/pick_orange_reference_100_v1.json`;
evaluators accept `EVAL_SEED_LIST_PATH`, `EVAL_RESULT_NAME`,
`ACTIONS_PER_CHUNK`, `MODEL_ID`, `N_INFERENCE_RUNS`, `MAX_STEPS`,
`EVAL_RESUME`, `EVAL_CHECKPOINT_PATH`, `EVAL_SUMMARY_PATH`, and
`SAVE_CAMERA_SNAPSHOTS` so queue jobs can be reproduced exactly.
After every `env.reset(seed=...)`, evaluators run two non-policy hold steps via
`refresh_observation_after_reset()` before saving start snapshots, auditing the
initial scene, or performing the first policy inference; immediate reset
observations and the first hold-step observation can contain the previous
episode's camera/COM buffers on the desktop stack.
Orchestrated `checkpoint.json` files include `trace_schema_version`
and per-episode story fields (`episode_summary`, `initial_scene`,
`final_scene`, `timeline`, `subtask_attempts`) so each run can be
reconstructed without console logs.

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

## Keeping this file current

Update this file **and** `AGENTS.md` in the same commit as any structural change: new scripts, deleted scripts, renamed files, or significant feature additions. The `.md` and the code must always agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**
