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
Target-switching via language is bounded by dataset coverage, not an intrinsic
VLA property. The policy only learned the language-to-target choice from high,
far-from-the-oranges configurations (seen at episode start and just after a
placement); grasps were collected one target at a time, so from states near an
orange it never saw a redirection branch and ignores a new prompt there. The
scripted spatial reset reproduces such a high/far configuration as a pragmatic
workaround (it also keeps the shared monotask data free of instruction-following
episodes). Do not write this up as a headline discovery.

---

# Report

The LaTeX toolchain and the figure/analysis scripts run here on the basement
desktop — the same machine that produces the inference and eval results.
(Historically the report was written on the Dell laptop; that split is gone.)

## Report state (June 2026)

Sections in `report/`:
- `introduction.tex` — draft
- `related_work.tex` — drafted; literature review (5 clusters: VLAs, IL failure modes, language/hierarchy decomposition, failure detection & recovery, self-improvement); cited via `references.bib`; included in `main.tex`
- `methodology.tex` — draft; Methods restructured — the dataset subsections are grouped under one `\subsection{Datasets}` (subsubsections: Public Baseline Dataset, Teleoperation, Autonomous Data Generation, Dataset Composition, Matched Monotask Comparison); the order is 3.1 Policy, 3.2 Task Decomposition, 3.3 Orchestrator, 3.4 Datasets. The Teleoperation subsubsection opens with the fixed-pick-order rationale (why language conditioning needs the custom dataset) and notes the abandoned Meta Quest 3 hand-tracking attempt (SO-101 has 5 arm joints vs the 6 DOF a full hand pose needs)
- `experiments.tex` — draft; full 100-episode seeded eval results in place. §4 = `4.1 Overall Performance` then `4.2 Failure Analysis` (subsubsections: The Grasp Bottleneck `sec:failure_modes`, Target Obedience `sec:obedience`)
- `results.tex` — draft; `\section{Discussion}` with `\subsection{Conclusion}` then `\subsection{Limitations}` (Conclusion first)
- `conclusion.tex` — draft (empty)
- `abstract.tex` — full abstract written (no longer a placeholder); also holds the project-pipeline overview figure (`fig:pipeline`, inline TikZ) placed after `\end{abstract}` so it renders **before** the Introduction. Sources: `Teleoperation (LeIsaac / Isaac Sim)` and `Public dataset (HuggingFace)` → **LeRobot** group (datasets → SmolVLA fine-tuning) → **Isaac Sim (LeIsaac)** eval box where the orchestrator prompts the SmolVLA policy, the policy drives the simulated robot, and the robot returns state; plus the autonomous-data feedback loop. (Stack is accurate: teleop is collected in Isaac Sim/LeIsaac as HDF5 then converted to LeRobot format; lerobot does training/inference, not collection.)
- `notes.md` — historical scratch pad; do not edit unless asked.
- `THESIS_BRIEF.md` — local copy of the official EPFL thesis requirements (20-page cap, prescribed structure); Notion source URL inside.

## Plotting scripts (`scripts/`)

Python scripts that generate the report figures. Run from `report/scripts/`.

| Script | Output figure |
|--------|---------------|
| `plot_baseline_comparison.py` | `figures/baseline_comparison.pdf` |
| `plot_data_effect.py` | `figures/data_effect.pdf` |
| `plot_orange_outcomes.py` | `figures/orange_outcome_distribution.pdf` |
| `plot_failure_modes.py` | `figures/failure_modes.pdf` — per-subtask outcome composition (GRASP/LIFT/PLACE × success/drop-slip/timeout) for the two **subtask** models; imports the composition helpers from `compute_failure_modes.py`. Pure-Python via `plot_lib`. |
| `plot_orchestrator_flow.py` | `figures/orchestrator_decision_flow.pdf` — decision-flow diagram. **Requires `matplotlib`** (the only script that does; the rest use the pure-Python `plot_lib` writer). |
| `plot_subtask_placeholders.py` | `figures/subtask_{grasp,lift,place}_placeholder.pdf` |
| `plot_dataset_composition.py` | `figures/dataset_composition.pdf` — subtask episode-length boxplot, Teleop vs Auto. Self-contained: stats baked in from `extract_dataset_composition.py`. |
| `extract_dataset_composition.py` | (no figure) Reads the LeRobot training datasets from the local HF cache and prints the per-source × per-subtask episode-length stats used by `plot_dataset_composition.py`. Needs the datasets + `pyarrow`/`numpy` (present here). |
| `extract_positional_prevalence.py` | (no figure) Prints the per-source count of GRASP episodes per positional label (left/right/middle/…). The `tab:label_prevalence` table it fed was **removed** — training-data composition is no longer mixed into the obedience results; kept for reference / the dataset section. Downloads only the small `meta/episodes` parquet from HF (Xet disabled); needs `huggingface_hub`/`pyarrow`. |
| `plot_grasp_confusion.py` | `figures/grasp_obedience_confusion.pdf` — 2×2 grid of grasp-obedience confusion matrices (requested vs. actually-grasped position) for the two subtask models × {0,1} oranges placed. Reads the git-tracked subtask checkpoints under `isaac-inference/results`; reconstructs grasp-time position labels. Pure-Python via `plot_lib`. |
| `compute_place_success.py` | (no figure) Prints place success for all four models (`tab:place_success`), read directly from the per-attempt `target_in_plate_end` field (eval schema EpisodeStory v2 / PhaseMonitor v3), with scene state from `scene_start.n_in_plate`. No anchoring/heuristics — the old gripper-retraction artifact is fixed at the source. Also prints the first-placement (empty-plate) column of `tab:place_success`; the old Level-1 place-recovery numbers are still printed but no longer tabulated. Self-validates `final_scene` vs `oranges_in_plate`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `compute_recovery.py` | (no figure) Prints leveled recovery for all four models. **Level-2** (different orange / task progress, `tab:task_recovery`): per-episode score recovery and per-event different-orange recovery after a LIFT drop / PLACE slip, measured identically for both formulations. (The **Level-1** same-orange "recovered" cells it also prints are no longer tabulated in the report — the `tab:lift_recovery`/`tab:place_recovery` recovery columns were removed in favour of the formulation-fair Level-2 view.) Reuses the same per-attempt fields as `compute_place_success.py`; self-validates `final_scene` vs `oranges_in_plate`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `compute_failure_modes.py` | (no figure) Prints, per model and subtask (**GRASP / LIFT / PLACE**), the genuine-attempt composition (success / drop-or-slip / timeout) with **geometric success override** — a timed-out attempt whose orange ended in the plate counts as success. GRASP has no drop/slip and is **subtask-only** (the monotask offline reconstruction over-segments the grasp search, so it has no per-attempt grasp rate). For the monotask (flat) models it also prints the **failure geometry** feeding `tab:failure_geometry`: PLACE release distance from the plate centre and LIFT height gain, success vs failure (median). Surfaces the tail-freeze timeout artefact. The `composition`/`grasp_composition` helpers are imported by `plot_failure_modes.py`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `compute_grasp_chain.py` | (no figure) Prints the grasp→place chain feeding `tab:grasp_chain`: subtask **grasp success rate**, **secured oranges/episode**, and **place once secured** (clean vs recovered), per model. Uses force-confirmed grasps + final positions only, so it is reliable for both formulations and **never prints a monotask grasp rate** (that trace is heuristic). Self-validates `final_scene` vs `oranges_in_plate`. Pure stdlib. |
| `compute_grasp_intent.py` | (no figure) Monotask grasp **intent** from the per-step columnar `geometry_trace` (eval schema v4): distinct oranges **tried**/episode (fills the `tab:grasp_chain` monotask `pending` cells), single-orange **fixation** distribution, and median closest grip-axis approach. "Tried" = nearest unplaced orange within `CENTER_M` of the grip axis for ≥ `MIN_CENTER_FRAMES` **consecutive** recorded frames (sustained centring — plain proximity over-counts, since oranges sit ~10 cm apart). Tunable thresholds, no eval rerun; degrades gracefully on pre-v4 checkpoints. Reads `flat_checkpoint.json` under `isaac-inference/results`; pure stdlib. |
| `compute_obedience.py` | (no figure) Prints **per-model** grasp obedience (overall + per spatial label) over scene states 0+1, reusing `plot_grasp_confusion.py`'s `confusion()` so it matches the figure's diagonal exactly (Teleop 67%, Teleop+Auto 73%; `right` worst at 41/47%). Also prints the share of `right` GRASP requests by `n_placed_start` (19→43→96%) — the plate-on-the-left geometry behind why `right` dominates. Feeds the **prose** of the obedience section (no table). Pure stdlib. |
| `plot_lib.py` | Shared helpers (colors, axis formatting, save wrappers) — imported by the scripts above; not run directly. |

**Note (results rewrite).** The results section is the merged **"The Grasp Bottleneck"** subsection (grasp composition figure via `plot_failure_modes.py`, GRASP strict force-confirmed ~37%) + one consolidated grasp-chain table + the obedience analysis. The table (grasp success, secured/ep, place-once-secured, oranges-tried) and all results figures/tables are now **caption-free** (the user's preference; referenced inline, no `\label`/`\cref`). Numbers come from `compute_grasp_chain.py`: per-formulation "secured" = force-grasp (subtask) / lift (monotask); Level-1 clean-vs-recovered; oranges-tried (subtask). The monotask oranges-tried cell is a bare `pending` placeholder. `compute_grasp_chain.py` also computes Level-2 redirection (GRASP-timeout→different-orange) and empty-episode rate, but those are **not shown** (dropped from the report). The old lift/place/recovery/failure-geometry tables are gone, so `compute_recovery.py`, the scene-state columns of `compute_place_success.py`, and the geometry parts of `compute_failure_modes.py` no longer feed report tables (its `composition` helper still feeds `plot_failure_modes.py`'s LIFT/PLACE bars). The **obedience** subsection is now prose + the confusion figure only (no prevalence/obedience table): per-model obedience (67/73%), the leftward misgrab bias (`right` worst), and the plate-left geometry (`right` requests 19→96% as oranges are placed), all from `compute_obedience.py`. The terms "grounding"/"coverage" were removed as unsupported buzzwords (adding Auto data slightly *raised* obedience).

## Figures (`figures/`)

PDF outputs generated by `scripts/`. Also contains `logo.png`. Committed to git alongside the LaTeX source. Two figures are **authored inline as TikZ** in the section sources (not generated by `scripts/`): the SmolVLA architecture (`fig:smolvla_finetune`, `methodology.tex`) and the project-pipeline overview (`fig:pipeline`, `abstract.tex`, rendered before the Introduction).

## Writing rules
- **HARD LIMIT: 20 pages maximum** for the compiled `main.pdf`. This is a firm
  constraint, not a target — the report must never exceed 20 pages. Condense wording
  and remove repetition before adding any length. Quality over quantity.
- Use the three-mechanism terminology consistently. If you find the
  word "recovery" used for spatial reset or redirection, flag it.
- The contribution framing is the orchestrator: two-level recovery
  (Level-1 local retry, Level-2 target redirection) wrapped around a VLA,
  with a scripted spatial reset that works around a coverage limit of the
  subtask dataset (target-switching was only demonstrated from high/far
  poses). Do NOT frame the spatial reset as a headline discovery or an
  intrinsic property of VLAs, and do NOT reduce the work to "we added
  recovery to a VLA."
- Every figure needs units, labels, and a message.
- No time-series plots unless unavoidable.
- Hypothesis-driven structure: the experiments in `results.tex` must
  map back to claims in the `introduction.tex`.
- A good report is not a list of things done; it is a structured
  reflection on them.
- **Captions are short.** A caption gives the figure's or table's single message; it does
  not restate axis labels, the legend, or what the body text already says.
- **Concise prose.** Remove any word that adds nothing; one clear message per paragraph,
  stated up front.
- **Avoid "we"** where possible; prefer the system or the passive as subject.
- **Signpost.** Open each section with a one-/two-sentence statement of what it covers.
- **Red thread.** Every paragraph, figure, and table must serve the central hypothesis ---
  no data dump of everything measured; details earn their place by adding to the contribution.
- **Figures:** units on both axes; show error bars / std where possible; pin floats (`[H]`,
  not drifting); put comparisons to prior work in a table; the final results figure should
  demonstrate the key hypothesis.
- **Conclusion = two paragraphs:** key findings/contributions, then future work.
- Full advice deck: `Master Thesis Thesis Writing.pdf` (repo root); also Whitesides, "Writing a Paper".

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

`references.bib` now holds the literature-review citations (21 entries cited via
`ieeetr`); `main.tex` wires `\bibliography`. The compiled report is ~18 pages
(hard limit 20), so keep an eye on length and trim repetition before adding more.

These come after the methodology rewrite and full eval results
land from the desktop.

## Reference
Thesis writing advice deck:
https://docs.google.com/presentation/d/1wdk6iVC3G0WgORenrWnNZ3Llxan1iO6rcQS7SlbRQxU/edit?usp=sharing

## Slides (`slides/`)

Oral-presentation deck, **LaTeX/Beamer** — `slides/slides.tex`. Built from the report
sections with the final seeded numbers; figures are pulled directly from
`report/figures/` (`\graphicspath{{../figures/}}`). The deck is **sequential**: every
concept (monotask, subtask, Teleop/Auto, Level-1/Level-2 recovery, spatial reset) is
defined before it is used. ~19 frames, 16:9.

Build:
```bash
export PATH="/home/students/texlive/2026/bin/x86_64-linux:$PATH"
cd report/slides && latexmk -pdf -interaction=nonstopmode -halt-on-error slides.tex
```
Output `slides/slides.pdf` is a build artifact (gitignored, like `main.pdf`); only
`slides.tex` is tracked. The previous PowerPoint pipeline (`presentation.pptx`,
`build_pptx.py`, `assets/`) has been removed in favour of this LaTeX deck.

## Keeping this file current

Update this file **and** `AGENTS.md` in the same commit as any structural change: new plotting scripts, new sections, renamed figures, updated section status. The `.md` and the code must always agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**