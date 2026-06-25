# MasterProject — Project Context

## What this project is
SmolVLA fine-tuned for pick-and-place on an SO-101 arm in Isaac Sim.
Task: pick 3 oranges, place in plate. VLA handles manipulation subtasks;
an algorithmic orchestrator sequences them using privileged Isaac state.

## Core conceptual framing (READ BEFORE EDITING ORCHESTRATOR OR REPORT)

Recovery is defined at TWO LEVELS of the goal hierarchy
(task > per-orange goal GRASP→LIFT→PLACE > subtask). Pinning every
recovery action to a level keeps the term precise and makes the
monotask-vs-subtask comparison fair. There are still THREE distinct
mechanisms — one recovery mechanism per level, plus an enabler. Do not
conflate them. Code, comments, and report text must use these names:

1. **Local retry** — LEVEL-1 (sub-goal) recovery; recovers the SAME
   orange. Orange slips during LIFT/PLACE, or LIFT/PLACE times out →
   return to GRASP for the SAME orange. Original sub-goal preserved.

2. **Target redirection** — LEVEL-2 (task) recovery; recovers task
   PROGRESS via a DIFFERENT orange. Repeated GRASP failure on orange A →
   give up on A, attempt orange B so the episode keeps scoring. The
   sub-goal is dropped, but the task objective (indifferent to WHICH
   orange is placed) is served — this is task-level recovery, NOT mere
   abandonment.

3. **Spatial reset** — a PRECONDITION / enabler, NOT a recovery level.
   VLA language conditioning only switches targets reliably when the arm
   is far from all oranges. Current implementation is fully scripted:
   first move only `shoulder_lift` toward the episode-start value, then
   move all joints to home. Required before any Level-2 target change.

Invariant: any change of language target must be preceded by a spatial
reset. A Level-1 retry on the same target after a slip does not.

The VLA is responsible for: GRASP(target), LIFT, PLACE.
The orchestrator is responsible for: subtask sequencing, outcome
classification, spatial resets, and target selection.

## Terminology — do not drift
- "Recovery" is TWO-LEVEL: Level-1 = local retry (same orange); Level-2
  = target redirection (different orange / task progress). Spatial reset
  is the enabler, NOT recovery — never call it "recovery".
- "Subtask" = one of GRASP / LIFT / PLACE. HOME is a scripted primitive,
  not a subtask.
- "Phantom grasp" = gripper closed on air; detected via gripper force.

## Key finding to preserve in writing
Target-switching via language is bounded by dataset coverage, not an intrinsic
VLA property. The policy only learned the language-to-target choice from high,
far-from-the-oranges configurations (seen at episode start and just after a
placement); grasps were collected one target at a time, so from states near an
orange it never saw a redirection branch and ignores a new prompt there. The
scripted spatial reset reproduces such a high/far configuration as a pragmatic
workaround (it also keeps the shared monotask data free of instruction-following
episodes). Do not write this up as a headline discovery.

---

# Isaac Inference (desktop)

Autonomous robot policy evaluation and dataset recording for the
MasterProject pick-and-place task, using Isaac Sim via LeIsaac and
the LeRobot framework.

## Status (June 2026)

Implementation and evaluation are complete. The orchestrator refactor, scripted spatial reset, full 100-episode seeded evals, and Phase 5 autonomous data generation are all done. Current focus is the report (`report/`).

## Directory layout

Runtime code stays at the `isaac-inference/` root; one-shot and support scripts are grouped into subfolders:

- **root** — inference/eval entry points (`inference_*.py`), shared libraries imported by them (`eval_utils.py`, `robot_utils.py`, `phase_monitor.py`, `dataset_recorder.py`), the eval-queue orchestrators (`overnight_eval_queue.py`, `rerun_gal_split_nolang_correct_prompt.py`), and `remote.sh`. These keep flat imports and are invoked by bare filename via `remote.sh`, so they are **not** moved.
- `dataset_pipeline/` — one-shot dataset build/transform scripts (run any of them with `python dataset_pipeline/<script>.py`; they resolve `synthetic_datasets/` at the root via `__file__`).
- `maintenance/` — diagnostic / repair / plotting one-shots.
- `tests/` — unit tests (add the root to `sys.path` via the shim at the top).
- `docs/` — reference notes (`commands.txt`, `EVAL_LOGGING_REWRITE.md`, and
  **`DATA_DICTIONARY.md`** — the source of truth for what every dataset/model name means;
  consult it before interpreting any `Gal-*` / `Gal_split*` artifact).
- Unchanged data/output dirs: `results/`, `logs/`, `eval_seeds/`, `legacy/`, `synthetic_datasets/`.

## Key files (current work)

| File | Purpose |
|------|---------|
| `inference_autonomous_orders.py` | **Main entry point** — orchestrator + eval loop. Target architecture: state machine over GRASP / LIFT / PLACE subtasks, with two-stage scripted GO_HOME reset and explicit RETRY_SAME / REDIRECT / ABANDON_ORANGE transitions. |
| `dataset_recorder.py` | Buffers frames per subtask; commits on success, discards on failure. Crash-safe via `checkpoint.json`. |
| `eval_utils.py` | `SubtaskTracker`, `EvaluationTracker`, `EpisodeStory`, `HomeChecker`, position/scene helpers. |
| `robot_utils.py` | Joint-space conversions: LeIsaac (radians) ↔ LeRobot (normalized degrees). |
| `remote.sh` | Isaac Sim launcher — sets `ENABLE_LIVESTREAM`, `LEISAAC_ASSETS_ROOT`, then calls `python "$@"`. |
| `docs/commands.txt` | Reference commands (training, inference, teleop, dataset conversion). |
| `inference_privileged_grasp.py` | Privileged grasp policy — uses DLS (Damped Least Squares) IK to position the gripper above each orange's exact XY from Isaac privileged state. Fully scripted/geometric; no VLA. |
| `inference_flat_prompt.py` | Flat single-instruction SmolVLA eval — one fixed prompt ("Place the orange into plate"), no subtask sequencing. Used to evaluate the no-lang-no-home model variant. |
| `inference_act_flat_prompt.py` | ACT policy eval — same flat single-instruction structure as `inference_flat_prompt.py` but uses `ACTPolicy` with chunked action execution (`predict_action_chunk`; chunk size comes from `policy.config.chunk_size`). Saves to `results/<model>/act_checkpoint.json` and `act_latest.txt`. |
| `phase_monitor.py` | `PhaseMonitor` — passive observer-inferred subtask trace for flat (no-language) policies, derived from physics state alone (`SubtaskTracker`, orange/plate positions). Used by `inference_flat_prompt.py` and `inference_act_flat_prompt.py` to produce per-episode subtask traces for policies that don't expose intent. |
| `tests/test_phase_monitor.py` | Unit tests for `PhaseMonitor` trace generation. |

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

One-shot scripts (in `dataset_pipeline/`) for building and transforming the training dataset. They resolve `synthetic_datasets/` at the `isaac-inference/` root via `__file__`, so they can be run from anywhere (e.g. `python dataset_pipeline/merge_datasets.py …`). Output goes to `synthetic_datasets/`.

| Script | Purpose |
|--------|---------|
| `dataset_pipeline/merge_datasets.py` | Concatenate two or more LeRobot v3.0 datasets into one. Episode indices are shifted and task strings are unioned. Re-encodes video to h264 (faster decode on non-RTX-40 hardware). Writes `meta/stats.json` (scalar features recomputed from the merged data parquet; image features count-weighted-aggregated from the sources). |
| `dataset_pipeline/balance_dataset.py` | Create a balanced subset across the 9 subtask slots (3 oranges × Grasp / Pick up / Place). N = min episode count per bucket; total output = 9×N. HOME episodes excluded. |
| `dataset_pipeline/consolidate_dataset_videos.py` | Merge per-episode chunk files into one large file per camera. Eliminates torchcodec decoder re-init overhead at training time. Idempotent — safe to re-run. |
| `dataset_pipeline/fix_merged_lengths.py` | Repair per-episode length drift that can appear after `consolidate_dataset_videos.py`. Run if episode lengths in `meta/episodes.parquet` look inconsistent. |
| `dataset_pipeline/tail_split.py` | Properly "tail" a teleop-derived dataset by **appending** `N=20` freeze frames to every episode (inverse of `strip_lang_and_tail.py`). Drops `"Go back to start position"` episodes and reproduces `dataset_recorder.py`'s per-subtask freeze: GRASP / LIFT hold the last commanded gripper action (closing force), PLACE freezes fully (action = state). Clones the last video frame (ffmpeg `tpad`), reconciles clip lengths, consolidates, writes `stats.json`. Used to rebuild `Gal_split` → `Gal_split_tailed` before merging into `Gal-merged-tailed-auto`. |
| `dataset_pipeline/strip_lang_and_tail.py` | Produce a no-language-conditioning clone with the last 20 frozen frames of every episode removed. Hardcoded paths: `Gal-merged-tailed-auto` → `Gal-merged-tailed-auto-no-lang-no-home`. |
| `dataset_pipeline/strip_tail.py` | Tail-only variant of the above: removes the 20 frozen tail frames per episode but **keeps the language labels**. Reads the HF-cache copy of `Gal_split_tailed`, writes `synthetic_datasets/Gal_split_notail` (846 eps, 87 966 frames). Used to train `Gal-pick-orange-notailCH20` for the tail-free ablation. |

## Maintenance / one-shot utilities

`maintenance/` holds diagnostic, repair, and plotting one-shots. The two eval-queue orchestrators stay at the root because they invoke the root entry points by bare filename via `remote.sh`.

| Script | Purpose |
|--------|---------|
| `maintenance/recover_episodes.py` | Repair corrupted episode parquet files. Originally written to fix `Gal-auto-subtasks2`. Run once as needed; not part of normal workflow. |
| `maintenance/debug_camera_drift.py` | Diagnostic: loads pick-orange env and runs 200 rapid resets, logging front camera world position before/after each `randomize_camera_uniform` call. Outputs `camera_drift_log_<hostname>.txt`. Run on both desktop and laptop to compare accumulation behaviour. |
| `maintenance/plot_dataset_stats.py` | Plot task distribution and composition statistics for a synthetic dataset. |
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

**Geometric outcome capture (schema: `EpisodeStory` v2, `PhaseMonitor` v3).**
Subtask attempts now carry `scene_start`/`scene_end` snapshots
(`scene_geometry()` in `eval_utils.py`: per-orange `in_plate`/position +
`n_in_plate`, tilt-aware and consistent with `oranges_in_plate`) and a derived
`target_in_plate_end`. This lets failures and timeouts be read directly from
"is the orange in the plate" for **both** monotask and subtask runs, replacing
brittle post-processing (the place gripper-retraction confirmation no longer
hides placements; placed-orange identities are explicit; no event/heuristic
reconstruction). Monotask has no subtask timeout (no per-subtask budget).
`final_scene` is now built from the same pre-reset snapshot used for the count
(fixing a bug where truncated episodes recorded post-auto-reset positions). These
are logging-only changes — control flow is unchanged, so a seeded rerun
reproduces identical trajectories with the richer fields. See
`docs/EVAL_LOGGING_REWRITE.md`. **The current committed checkpoints predate this and
must be re-run to populate the new fields.**

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
