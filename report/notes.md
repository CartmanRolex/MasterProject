# Project Notes — Enabling Recovery in VLA-Based Robotic Manipulation

Internal scratch pad. Do not copy directly into the report.

---

## Setup

- **Robot:** SO-101 arm (6 DOF)
- **Environment:** Kitchen scene in Isaac Sim — privileged state information available (joint positions, gripper force, object positions)
- **Task:** Pick 3 oranges from the scene and place them in a plate
- **Base policy:** SmolVLA, fine-tuned per experiment
- **HuggingFace org:** `MasterProject2026`

---

## Code Paths

| Component | File |
|-----------|------|
| Main orchestrator / autonomous eval loop | `isaac-inference/inference_autonomous_orders.py` |
| Subtask dataset recorder (crash-safe) | `isaac-inference/dataset_recorder.py` |
| SubtaskTracker, EvaluationTracker, HomeChecker | `isaac-inference/eval_utils.py` |
| Joint-space conversion (LeIsaac rad ↔ LeRobot deg) | `isaac-inference/robot_utils.py` |
| Eval result logs + plot script | `isaac-inference/results/` |
| Xbox gamepad teleoperation (v3, current) | `leisaac-mods/so101_gamepad_v3.py` |
| Dataset annotation GUI (frame-by-frame editor) | `dataset-editor/editor.py` |
| Trailing-frame fixer for Place episodes | `dataset-editor/tailer.py` |
| SLURM training scripts (1 GPU / 4 GPU) | `cluster-training/train.sh`, `cluster-training/train_xl.sh` |
| Determinism / state-checkpoint test | `isaac-inference/inference_explore.py` |

---

## Subtask Phases & Prompts

The autonomous orchestrator sequences four phases per orange, each with a step timeout:

| Phase | Prompt sent to SmolVLA | Timeout (steps) |
|-------|------------------------|-----------------|
| GRASP | `"Grasp <left/middle/right> orange"` | 700 |
| LIFT | `"Pick it up"` | 400 |
| PLACE | `"Place it into plate"` | 500 |

Spatial reset (HOME) is **not a VLA subtask**. It is a scripted joint-space interpolation to the known home configuration, executed deterministically by the orchestrator. It is a precondition for target changes, not a recovery mechanism. The old `"Go back to start position"` VLA prompt (150 steps) has been superseded by this scripted controller.

VLA prompts are defined in `inference_autonomous_orders.py::build_task_prompt()`.

### Freeze frames
At the end of every recorded subtask episode, **20 freeze frames** are appended so the policy has a clear "hold this pose" signal at the boundary. The freeze action differs by phase (`dataset_recorder.py` lines 217–241):

- **GRASP and LIFT** (`"Grasp …"` / `"Pick it up"`): arm joints freeze at their current state, but the **gripper joint keeps its last commanded action value** — this maintains the closing force on the orange (action ≠ state here because the gripper is still being driven closed).
- **All other phases** (PLACE, HOME, RECOVERY): `action = observation.state`, commanding "stay exactly here" with no net movement.

---

## Phase 1 — Baseline fine-tuning on public dataset

- **Dataset:** `LightwheelAI/leisaac-pick-orange-mimic-v0` — 60 episodes, each a full 3-orange pick-and-place sequence (monolithic, no subtask structure).
- **Models trained:**
  - SmolVLA → `MasterProject2026/pick-orange-mimic`
  - ACT → `MasterProject2026/ACT-pick-orange`
- **Measured success rates (100-episode eval, full task):**
  - ACT: **26/100 (26%)**
  - SmolVLA (checkpoint 035000): **20/100 (20%)**; (checkpoint 040000): **28/100 (28%)**
- **Training hyperparameters:**
  - SmolVLA: batch_size=32, steps=40000, chunk_size=20, n_action_steps=20
  - ACT: batch_size=8, steps=100000
- **Takeaway:** Near-identical performance between ACT and SmolVLA suggests the VLA's language-conditioning capability is not being exploited under a monolithic full-task framing. Scaling the policy class alone yields no gain.

---

## Phase 2 — Subtask decomposition on public dataset

- **Motivation:** Decompose into language-specified subtasks so a higher-level orchestrator can sequence them and interrupt on failure. Key failure mode: *phantom grasp* — gripper closes on air, then proceeds to PLACE as if holding an orange.
- **Method:** Re-sliced 60 episodes → ~600 subtask episodes. Retrained SmolVLA.
- **Result:** No clear qualitative improvement. No formal success rate measured — subtask mode is significantly slower per rollout, making large-scale eval expensive for what appeared by eye to be no real gain.
- **Diagnosed limitation:** The public dataset always picks oranges in the **same order** → no signal tying a language instruction to a specific orange. The policy cannot be conditioned to target a particular orange, which is a prerequisite for the orchestrator-driven recovery envisioned later.

---

## Phase 3 — Custom teleoperated dataset

- **Motivation:** Build a dataset that explicitly supports (1) targeted-grasp conditioning by relative position, (2) subtask-level episodes from the start, (3) diverse pick orders.

### Teleoperation pipeline
- Built from scratch: Xbox controller → SO-101 arm in Isaac kitchen scene.
- Implementation: `leisaac-mods/so101_gamepad_v3.py` (v3 is current; v2 superseded). Roll-lock toggle: press **X** on the controller.
- Substantial engineering deliverable in its own right.

### Dataset design
- Each episode = **one subtask** (not full task).
  - Advantages: easy to exclude failed teleop attempts; easy to annotate with relative-position instructions.
  - Instructions: `"Grasp left orange"`, `"Grasp middle orange"`, `"Grasp right orange"`, `"Place it into plate"`, `"Go back to start position"`.
- Pick order randomized across episodes → breaks spurious order correlation from public dataset.
- Trailing frozen frames fixed with `dataset-editor/tailer.py` before training.
- **"tailed"** in the model name = dataset processed with `tailer.py`; **"CH20"** = chunk_size / n_action_steps = 20.
- **Final size: 921 subtask episodes.**

### Model
- `MasterProject2026/Gal-pick-orange-tailedCH20`

### Result
- SmolVLA fine-tuned on this dataset follows relative-position instructions and can target a specific orange.
- **Full-task standalone eval (no orchestrator): 0–1/100** — expected, because this is a subtask model not designed to run monolithically.
- With orchestrator: formal evaluation is worse than the monolithic baselines. The policy can still follow target instructions qualitatively, but this is not enough to offset low grasp reliability under full-task execution.
- Formal orchestrated eval: **26/326 (7.98%)**, average **1.32/3** oranges in plate, stopped after 326/500 planned episodes.

---

## Key VLA Behavioral Finding

**VLA language conditioning is spatially context-dependent.**

When the gripper is close to orange A (during or after a failed grasp attempt), issuing a language instruction targeting a different orange ("Grasp right orange") is ignored — the VLA remains focused on the nearby orange. Language-based target switching only works reliably when the arm is spatially far from all oranges (i.e., near the start/home configuration).

This was discovered empirically: after a GRASP timeout, issuing a new target prompt without first moving the arm away consistently fails.

**Implemented design — scripted spatial reset:**

The early "Go back to start position" VLA prompt conflated two purposes: (1) sim episode-end constraint, and (2) language conditioning precondition. The VLA was unreliable for this because it was asked to return to home from out-of-distribution post-failure poses, and 150 steps was often insufficient.

The implemented fix replaces this with a scripted joint-space interpolation to the known home configuration. The VLA is responsible for exactly three subtasks: GRASP, LIFT, PLACE. Spatial resets are handled deterministically by the orchestrator. This is a principled design decision: before issuing any new target instruction, the orchestrator guarantees the arm is in a state where language conditioning is reliable.

Terminology (do not conflate):
- **Local retry** = true recovery; retries same orange after a slip; no spatial reset needed
- **Spatial reset** = scripted HOME interpolation; precondition for any target change; not recovery
- **Target redirection** = abandons current orange, pivots to different one; not recovery

---

## Phase 4 — Orchestrator and success detection (current state)

- **Implementation:** Algorithmic orchestrator reading **privileged Isaac state** — not a VLM. Detects: gripper occupancy post-grasp (force vectors), orange-in-plate post-placement (position check). Can later be replaced by a VLM, but privileged state avoids introducing perception noise at this stage.
- **Key code:** `isaac-inference/eval_utils.py` (SubtaskTracker, EvaluationTracker, HomeChecker) + `inference_autonomous_orders.py` (OrderController class).
- **Capabilities:**
  1. **Phantom-grasp detection:** prevents advancing to LIFT/PLACE if gripper is empty after GRASP.
  2. **Targeted redirection:** if a specific orange is repeatedly ungraspable (GRASP timeout), re-issue instruction targeting a different orange.
- **Important framing:** Redirecting to a *different* orange is **not recovery** of the original goal. It is (a) a mechanism to avoid getting stuck and improve overall task completion rate, and (b) a key enabler for diverse autonomous data collection in Phase 5.

---

## Eval Infrastructure Updates (May 2026)

- `EvaluationTracker` now supports checkpoint/resume: results persist across interruptions in `results/eval_<model>_checkpoint.json`.
- Per-subtask success rates are now tracked broken down by how many oranges are already placed at subtask start (e.g. GRASP with 0 placed vs 1 placed vs 2 placed).
- Eval and data recording are controlled independently: `RECORD_ENABLED` flag. These should be run separately — eval first (recording off), then data collection (recording on).
- Current dataset target: `Gal-auto-subtasks3`.

### Partial eval results — `Gal-pick-orange-tailedCH20` (326/500 runs, May 2026)

```
Success Rate:         26/326 (7.98%)
Avg oranges in plate: 1.32/3
3/3 oranges:          26/326 (8.0%)
2/3 oranges:          121/326 (37.1%)
1/3 oranges:          110/326 (33.7%)
0/3 oranges:          69/326 (21.2%)

GRASP:  0 placed: 357/921 (38.8%)   1 placed: 204/608 (33.6%)   2 placed: 51/195 (26.2%)
LIFT:   0 placed: 302/357 (84.6%)   1 placed: 174/204 (85.3%)   2 placed: 39/51  (76.5%)
PLACE:  0 placed: 237/302 (78.5%)   1 placed: 140/174 (80.5%)   2 placed: 30/39  (76.9%)

Local retries:        791 total, fired in 314/326 episodes
Target redirections:  523 total, fired in 293/326 episodes
Oranges abandoned:    515 total (1.58 avg/episode)
```

**Key takeaway:** The new formal result is **worse** than the monolithic baselines (ACT 26%, SmolVLA 28%). LIFT and PLACE are comparatively reliable once the robot has the orange, but GRASP remains the bottleneck and degrades as the episode progresses. The orchestrator fires local retry and redirection frequently, but these mechanisms do not compensate for the weak grasp policy strongly enough to improve full-task success.

---

## Phase 5 — Autonomous data generation (next step)

- **Motivation:** Manual teleoperation is the bottleneck on dataset size and diversity. The current targeted-grasp model + orchestrator can run fully autonomous rollouts.
- **Recording target:** `MasterProject2026/Gal-auto-subtasks2` (configured in `inference_autonomous_orders.py` as `RECORD_DATASET_NAME`).
- **Pipeline:** Fully autonomous. Isaac privileged state labels each subtask attempt as success/failure. Only **successful** subtask trajectories are stored (`dataset_recorder.py` → commit on success, discard on failure).
- **Crash safety:** `checkpoint.json` tracks total committed recordings; parquet writer is closed after every commit so a hard crash leaves valid data.
- **Why redirection matters here:** Without targeted-grasp control, an ungraspable orange forces a full episode reset (wasted configuration). With redirection, the system pivots to a different orange in the same scene — extracting useful successful subtask data from configurations that would otherwise be discarded.
- **Goal:** Use auto-generated data to retrain SmolVLA for better generalization to OOD configurations, without further manual annotation.

---

## Deliverables to Date

1. Xbox-controller teleoperation pipeline for SO-101 in Isaac (`leisaac-mods/so101_gamepad_v3.py`), with subtask segmentation and relative-position annotation.
2. 921-episode subtask-level dataset (`MasterProject2026/Gal-pick-orange-tailedCH20` training data) with relative-position language annotations and randomized pick orders.
3. SmolVLA fine-tuned to follow relative-position instructions (`MasterProject2026/Gal-pick-orange-tailedCH20`).
4. Algorithmic orchestrator (Isaac privileged state) that sequences subtasks, detects phantom grasps, and redirects targeting (`isaac-inference/inference_autonomous_orders.py` + `eval_utils.py`).
5. Fully autonomous data-generation pipeline that filters for successful subtasks (`isaac-inference/dataset_recorder.py`).
