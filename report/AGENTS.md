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
   is far from all oranges. Implemented as a SCRIPTED joint-space
   interpolation to home (NOT a VLA prompt). Required before any Level-2
   target change.

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
VLA language conditioning is spatially context-dependent. Target
switching fails when the gripper is near a non-target orange. This
motivates the scripted spatial reset.

---

# Report (laptop)

LaTeX toolchain lives on this laptop. Inference and eval results are
produced on the desktop and synced via git. Do not attempt to run
inference from this machine.

## Report state (June 2026)

Sections in `report/`:
- `introduction.tex` — draft
- `related_work.tex` — draft (empty)
- `methodology.tex` — draft; terminology now uses the three-mechanism framing but needs a full structural rewrite
- `experiments.tex` — draft; full 100-episode seeded eval results in place
- `results.tex` — draft (Discussion & Conclusion section)
- `conclusion.tex` — draft (empty)
- `abstract.tex` — draft, write last
- `notes.md` — historical scratch pad; do not edit unless asked.

## Plotting scripts (`scripts/`)

Python scripts that generate the report figures. Run from `report/scripts/` on the laptop.

| Script | Output figure |
|--------|---------------|
| `plot_baseline_comparison.py` | `figures/baseline_comparison.pdf` |
| `plot_data_effect.py` | `figures/data_effect.pdf` |
| `plot_orange_outcomes.py` | `figures/orange_outcome_distribution.pdf` |
| `plot_orchestrator_flow.py` | `figures/orchestrator_decision_flow.pdf` — decision-flow diagram. **Requires `matplotlib`** (the only script that does; the rest use the pure-Python `plot_lib` writer). |
| `plot_subtask_placeholders.py` | `figures/subtask_{grasp,lift,place}_placeholder.pdf` |
| `plot_dataset_composition.py` | `figures/dataset_composition.pdf` — subtask episode-length boxplot, Teleop vs Auto. Self-contained: stats baked in from `extract_dataset_composition.py`. |
| `extract_dataset_composition.py` | (no figure) Reads the LeRobot training datasets from the desktop HF cache and prints the per-source × per-subtask episode-length stats used by `plot_dataset_composition.py`. **Desktop only** — needs the datasets + `pyarrow`/`numpy`. |
| `extract_positional_prevalence.py` | (no figure) Prints the per-source count of GRASP episodes per positional label (left/right/middle/…) for the `tab:label_prevalence` table in `experiments.tex`. Downloads only the small `meta/episodes` parquet from HF (Xet disabled), so it runs on the laptop; needs `huggingface_hub`/`pyarrow`. |
| `plot_grasp_confusion.py` | `figures/grasp_obedience_confusion.pdf` — 2×2 grid of grasp-obedience confusion matrices (requested vs. actually-grasped position) for the two subtask models × {0,1} oranges placed. Reads the git-tracked subtask checkpoints under `isaac-inference/results`; reconstructs grasp-time position labels. Pure-Python via `plot_lib`. |
| `compute_place_success.py` | (no figure) Prints place success for all four models (`tab:place_success`), read directly from the per-attempt `target_in_plate_end` field (eval schema EpisodeStory v2 / PhaseMonitor v3), with scene state from `scene_start.n_in_plate`. No anchoring/heuristics — the old gripper-retraction artifact is fixed at the source. Also prints post-failure place recovery / re-engagement (`tab:place_recovery`). Self-validates `final_scene` vs `oranges_in_plate`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `compute_recovery.py` | (no figure) Prints leveled recovery for all four models. **Level-1** (same orange): lift + place reclaim — reproduces `tab:lift_recovery` / `tab:place_recovery` "recovered" cells. **Level-2** (different orange / task progress, `tab:task_recovery`): per-episode score recovery and per-event different-orange recovery after a LIFT drop / PLACE slip, measured identically for both formulations. Reuses the same per-attempt fields as `compute_place_success.py`; self-validates `final_scene` vs `oranges_in_plate`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `plot_lib.py` | Shared helpers (colors, axis formatting, save wrappers) — imported by the scripts above; not run directly. |

## Figures (`figures/`)

PDF outputs generated by `scripts/`. Also contains `logo.png`. Committed to git alongside the LaTeX source.

## Writing rules
- Max 20 pages. Quality over quantity.
- Use the three-mechanism terminology consistently. If you find the
  word "recovery" used for spatial reset or redirection, flag it.
- The contribution framing is: "VLA language conditioning has a
  spatial precondition; we built an orchestrator that respects it,
  with scripted spatial reset and explicit target redirection."
  NOT: "we added recovery to a VLA."
- Every figure needs units, labels, and a message.
- No time-series plots unless unavoidable.
- Hypothesis-driven structure: the experiments in `results.tex` must
  map back to claims in the `introduction.tex`.
- A good report is not a list of things done; it is a structured
  reflection on them.

## Structure (per assignment brief)

- **Introduction** — problem, goal (with related SOTA), hypothesis,
  what is demonstrated, key contributions. Optional Problem Statement
  sub-section.
- **Methods** — methods developed, simulation / model / fabrication
  formulation. Figures welcome.
- **Results** — experimental results that map back to the hypothesis.
  Avoid time-series unless unavoidable.
- **Discussion & Conclusion** — what was learnt, how it answers the
  hypothesis, future extensions.
- **References**
- **Supplementary Information**

## Out of scope right now
- Supplementary material
- Polishing the abstract
- Bibliography formatting

All three come after the methodology rewrite and full eval results
land from the desktop.

## Reference
Thesis writing advice deck:
https://docs.google.com/presentation/d/1wdk6iVC3G0WgORenrWnNZ3Llxan1iO6rcQS7SlbRQxU/edit?usp=sharing

## Keeping this file current

Update this file **and** `CLAUDE.md` in the same commit as any structural change: new plotting scripts, new sections, renamed figures, updated section status. The `.md` and the code must always agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**
