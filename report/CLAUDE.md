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
- **Report surface terms (unified):** the non-decomposed baseline is the
  **monotask** formulation (never "monolithic"), and the spatial-reset enabler
  is called the **home-pose reset** in the report prose (same mechanism as
  "Spatial reset" above; code may still print "spatial reset").

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
- `methodology.tex` — draft; Methods restructured — the dataset subsections are grouped under one `\subsection{Datasets}` (subsubsections: Public Baseline Dataset, Teleoperation, Autonomous Data Generation, Dataset Composition, Matched Monotask Comparison); the order is 3.1 Policy, 3.2 Task Decomposition, 3.3 Orchestrator, 3.4 Datasets. The Teleoperation subsubsection opens with the fixed-pick-order rationale (why language conditioning needs the custom dataset); the abandoned Meta Quest 3 hand-tracking attempt (SO-101 has 5 arm joints vs the 6 DOF a full hand pose needs; pipeline built late, no data collected) is mentioned only in Future Work
- `experiments.tex` — draft; full 100-episode seeded eval results in place. §4 = `4.1 Overall Performance` then `4.2 Failure Analysis` (subsubsections: Grasp Acquisition and Recovery Behaviour `sec:failure_modes`, Target Obedience `sec:obedience`)
- `results.tex` — draft; `\section{Discussion}` with `\subsection{Conclusion}`, `\subsection{Limitations}`, then `\subsection{Future Work}` split into two subsubsections: `Subtask Execution` (grasp/target policy via RL or targeted demos; a 6-DOF arm for easier teleop / better data) and `Orchestration` (remove privileged success checks via a VLM monitor; more complex tasks where an FSM or external LM/VLM selects subtasks)
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
| `plot_orange_outcomes.py` | `figures/orange_outcome_recipes.pdf` — §4.1's single results figure: three stacked panels, one per fine-tuning recipe (standard / partial=partial / full=full), the five model families in fixed columns (LightwheelAI · Teleop mono/sub · Teleop+Auto mono/sub), one shared legend, batch size + steps annotated per panel (control excepted per bar). Standard-panel baseline bar = `pick-orange-mimic-b64s20k` (regime-matched retrain, 43%/2.06; upstream 41% model kept as full-recipe regime anchor). Full recipe improves every family (36–61%; best = baseline full 61%/2.20); panel 2 carries the batch32/40k **regime control** (`…-unfrozen-vlm-b32s40k`, 40%/2.03). No-tail dropped from the figure (inline numbers only). **Also** `figures/orange_outcome_standard.pdf` — **no longer referenced** (the slides now use `orange_outcome_recipes.pdf`); `orange_outcome_variants.pdf` removed. Pure-Python via `plot_lib` (`draw_recipe_panels`). |
| `plot_failure_modes.py` | `figures/failure_modes.pdf` — per-subtask outcome composition (GRASP/LIFT/PLACE × success/drop-slip/timeout) for the two **subtask** models, **standard variant only**; imports the composition helpers from `compute_failure_modes.py`. **No longer referenced** — report §4.2.1 and the slides both use the per-variant figures below; kept for reference. Pure-Python via `plot_lib`. |
| `plot_failure_modes_variants.py` | `figures/failure_modes_variants_subtask.pdf` + `figures/failure_modes_variants_monotask.pdf` — §4.2.1's two per-subtask composition figures across all three fine-tuning **variants** (standard/partial/full). One panel per dataset (Teleop, Teleop+Auto); GRASP/LIFT/PLACE grouped, three variant bars each, success/drop-slip/timeout share with a `%` x-axis. **Subtask** figure = GRASP/LIFT/PLACE; **monotask** figure = LIFT/PLACE only (per-attempt grasp not inferable) and drops the Timeout legend (its offline reconstruction expresses failures as drop/slip only). Reuses `grasp_strict` + colours from `plot_failure_modes.py` and `composition` from `compute_failure_modes.py`. Pure-Python via `plot_lib`. |
| `plot_orchestrator_flow.py` | `figures/orchestrator_decision_flow.pdf` — decision-flow diagram. **Requires `matplotlib`** (the only script that does; the rest use the pure-Python `plot_lib` writer). |
| `plot_subtask_placeholders.py` | `figures/subtask_{grasp,lift,place}_placeholder.pdf` |
| `plot_dataset_composition.py` | `figures/dataset_composition.pdf` — subtask episode-length boxplot, Teleop vs Auto. Self-contained: stats baked in from `extract_dataset_composition.py`. |
| `extract_dataset_composition.py` | (no figure) Reads the LeRobot training datasets from the local HF cache and prints the per-source × per-subtask episode-length stats used by `plot_dataset_composition.py`. Needs the datasets + `pyarrow`/`numpy` (present here). |
| `extract_positional_prevalence.py` | (no figure) Prints the per-source count of GRASP episodes per positional label (left/right/middle/…). The `tab:label_prevalence` table it fed was **removed** — training-data composition is no longer mixed into the obedience results; kept for reference / the dataset section. Downloads only the small `meta/episodes` parquet from HF (Xet disabled); needs `huggingface_hub`/`pyarrow`. |
| `plot_grasp_confusion.py` | `figures/grasp_obedience_confusion.pdf` — 2×3 grid of grasp-obedience confusion matrices (requested vs. actually-grasped position) for the **six subtask models** (rows Teleop/Teleop+Auto × columns standard/partial/full; all cells filled, legend in a strip along the bottom), scene states 0--1 **pooled**, cells = % of the row's grasps, position labels written out in full, explicit "Requested orange"/"Grasped orange" axis titles per band (no subtitle text). Shows that training more of the VLM does not improve obedience + the leftward-misgrab structure. Reads the git-tracked subtask checkpoints under `isaac-inference/results`; reconstructs grasp-time position labels. Exports `GRID`/`RECIPES`/`confusion()` used by `compute_obedience.py`. Pure-Python via `plot_lib`. |
| `compute_place_success.py` | (no figure) Prints place success for all four models (`tab:place_success`), read directly from the per-attempt `target_in_plate_end` field (eval schema EpisodeStory v2 / PhaseMonitor v3), with scene state from `scene_start.n_in_plate`. No anchoring/heuristics — the old gripper-retraction artifact is fixed at the source. Also prints the first-placement (empty-plate) column of `tab:place_success`; the old Level-1 place-recovery numbers are still printed but no longer tabulated. Self-validates `final_scene` vs `oranges_in_plate`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `compute_recovery.py` | (no figure) Prints leveled recovery for all four models. **Level-2** (different orange / task progress, `tab:task_recovery`): per-episode score recovery and per-event different-orange recovery after a LIFT drop / PLACE slip, measured identically for both formulations. (The **Level-1** same-orange "recovered" cells it also prints are no longer tabulated in the report — the `tab:lift_recovery`/`tab:place_recovery` recovery columns were removed in favour of the formulation-fair Level-2 view.) Reuses the same per-attempt fields as `compute_place_success.py`; self-validates `final_scene` vs `oranges_in_plate`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `compute_failure_modes.py` | (no figure) Prints, per model and subtask (**GRASP / LIFT / PLACE**), the genuine-attempt composition (success / drop-or-slip / timeout) with **geometric success override** — a timed-out attempt whose orange ended in the plate counts as success. GRASP has no drop/slip and is **subtask-only** (the monotask offline reconstruction over-segments the grasp search, so it has no per-attempt grasp rate). For the monotask (flat) models it also prints the **failure geometry** feeding `tab:failure_geometry`: PLACE release distance from the plate centre and LIFT height gain, success vs failure (median). Surfaces the tail-freeze timeout artefact. The `composition`/`grasp_composition` helpers are imported by `plot_failure_modes.py`. Reads checkpoints under `isaac-inference/results`; pure stdlib. |
| `compute_grasp_chain.py` | (no figure) Prints the grasp→place chain: subtask **grasp success rate** (~37%), **secured oranges/episode**, and **place once secured** split clean vs **recovered** (= Level-1 re-grasps, ~6–11%). The §4.2.1 grasp-chain *table* was removed; these numbers are now cited inline. Uses force-confirmed grasps + final positions; **never prints a monotask grasp rate** (not comparable to the budgeted subtask attempts). Self-validates `final_scene` vs `oranges_in_plate`. Pure stdlib. |
| `compute_level2_deadline.py` | (no figure) Prints the **deadline-anchored Level-2 comparison** feeding §4.2.1's `tab:level2_deadline` + prose. A **stall** = one grasp budget (700 steps) of consecutive fruitless commitment to one orange: subtask = first timed-out GRASP prompt in a same-target run (exact); monotask = 600 trace steps with the same nearest-unplaced orange and no force-confirmed grasp (600 not 700: calibrated ~130–150-step head start, successful acquisition medians 286–290 from home vs 140–170 from nearest). Opportunity-gated (≥1 *other* orange unplaced) and censored (report level: ≥1500 steps left, > median steps-to-place). Per stall: **redirected** (next orange differs), **placed-after** split by identity of the first post-deadline placement (**same vs different** orange), **steps-to-place**; also prints the redirect/persist response split and sensitivity sweeps (censor 0/500/1000, threshold 500/700). Self-validates `final_scene` vs `oranges_in_plate`. Pure stdlib. |
| `plot_recovery_outcomes.py` | **No longer feeds the report** — superseded by `compute_level2_deadline.py` (its first-orange anchoring was asymmetric: subtask = first GRASP prompt vs monotask = first force-confirmed grasp). Kept for reference; `figures/recovery_outcomes.pdf` was deleted. |
| `compute_composite_labels.py` | (no figure) Prints the top-right/bottom-right pair geometry for composite-label GRASP requests (five subtask models pooled, states 0+1): lateral and depth gap, obeyed vs within-pair misgrab. Feeds the "Inconsistent composite-label boundaries" **Limitations item** — misgrabs do not cluster near the 3 cm `ORANGE_AXIS_TOL` boundary (same x-gap distribution, median 2.1 cm) and the pair is always depth-separated (>=5 cm), so the mixups read as a supervision gap, not ambiguous eval scenes. Pure stdlib. |
| `compute_obedience.py` | (no figure) Prints **per-model** grasp obedience (overall + per spatial label) over scene states 0+1, reusing `plot_grasp_confusion.py`'s `confusion()` so it matches the figure's diagonal exactly, for all **six** subtask models (standard 67/73%, partial 67/67%, full 64/64% — training more of the VLM does not improve obedience; `right` worst at 27--47%). Feeds the **prose** of §4.2.2 (no table). Pure stdlib. |
| `plot_lib.py` | Shared helpers (colors, axis formatting, save wrappers) — imported by the scripts above; not run directly. |

**Note (results rewrite).** §4.2.1 (Grasp Acquisition and Recovery) = the two per-variant grasp-bottleneck composition figures (`plot_failure_modes_variants.py`, subtask + monotask; GRASP strict force-confirmed ~37% on standard) + the **deadline-anchored Level-2 comparison** (`compute_level2_deadline.py` → `tab:level2_deadline`, the report's only table) + the obedience analysis (§4.2.2). The old grasp-chain **table**, the **"oranges tried"** metric, and the **first-orange outcome figure** (`plot_recovery_outcomes.py`, asymmetric anchoring) were all **removed**. §4.2.1's recovery block: stalls (see script row above), redirected 100% subtask vs ~45–54% monotask, placed-after 71%/43% vs 31%/34% with the whole gap in the **different-orange** component (53%/34% vs 16%/17%) — the Level-2 claim; the same-orange component is formulation-independent (~10–18%). Level-1's marginal contribution (~6–11% of placements, from `compute_grasp_chain.py`) stays inline. Only the redirection *behaviour* is attributed to the recovery logic; conversion rates stay **outcome-level** (per-attempt grasp quality also contributes). Results figures are referenced inline with no `\label`/`\cref`; the two §4.2.1 composition figures carry short descriptive captions, the rest are caption-free, and the deadline table has a short caption + `tab:level2_deadline`. The old lift/place/recovery/failure-geometry tables are gone, so `compute_recovery.py`, the scene-state columns of `compute_place_success.py`, and the geometry parts of `compute_failure_modes.py` no longer feed report tables (its `composition` helper still feeds `plot_failure_modes.py`'s LIFT/PLACE bars). §4.2.2 (**Grasp Obedience**) is 2 short paragraphs + the confusion figure (now **6 subtask models** standard/partial/full in a full 2×3 grid, scene states pooled): obedience 64--73\% and **training more of the VLM does not improve it** (67% on both partial variants, 64% on both full — despite the full models being the best overall); the leftward misgrab bias (`right` worst, 27--47%); plus a third paragraph on the asymmetric bottom-right->top-right confusion (29--62% one way, 0--9% the other), whose labeling-inconsistency hypothesis lives as a **Limitations item**, not in 4.2.2: teleop labels typed by eye vs the scripted 3 cm `ORANGE_AXIS_TOL` used by auto data + eval, so conventions disagree *within* the merged training set and between training and eval (a rule-labeled top/bottom-right pair may have been annotated right/middle by eye); only 16 bottom-right grasp demos; `compute_composite_labels.py` shows the eval mixups do NOT cluster near the boundary and pairs are depth-separated — supervision gap, not ambiguous scenes. The hand-typed teleop labels were never audited against the rule (no scene geometry here). The scene-state right-share (19→43→96%) and the "weakest instruction" paragraph were dropped. Numbers from `compute_obedience.py`. The terms "grounding"/"coverage" were removed as unsupported buzzwords.

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

`references.bib` now holds the literature-review citations (25 entries cited via
`ieeetr`); `main.tex` wires `\bibliography`. The compiled report is ~19 numbered pages
(hard limit 20), so keep an eye on length and trim repetition before adding more.

These come after the methodology rewrite and full eval results
land from the desktop.

## Reference
Thesis writing advice deck:
https://docs.google.com/presentation/d/1wdk6iVC3G0WgORenrWnNZ3Llxan1iO6rcQS7SlbRQxU/edit?usp=sharing

## Slides (`slides/`)

Oral-presentation deck, **LaTeX/Beamer** — `slides/slides.tex`. Built for a **20-minute
talk**: **figure-driven and keyword-sparse** (the presenter narrates; each content slide
carries a few keywords + one figure/result, not full sentences). Figures are pulled
directly from `report/figures/` (`\graphicspath{{../figures/}}`) and the deck **reuses the
report's own figures** — all models, not slides-only simplifications. Bullets are
**keyword fragments, not sentences** (the presenter develops them orally). **15 frames**
(13 main + 2 Q&A backup), 16:9, sequential (every concept defined before use). The
narrative order is **user-prescribed — do not reorder without asking**: the problem stated
generally (IL limited, OOD, goal = recover; illustrated by a small inline OOD-drift TikZ
sketch) → the idea (VLA as **low-level controller**, a high-level controller interacting
**via text**; carries the project-pipeline TikZ ported from `abstract.tex`) → environment
& task (5 joints + 1 gripper; per-episode randomisation of top-camera pose + orange
positions; env figures side by side) → **decomposition & data merged on one slide**
(GRASP/LIFT/PLACE + by-position; then the 3 sources LightwheelAI/Teleop/Auto with
end-to-end vs subtask episode counts spelled out; one **monotask** bullet = "same
trajectories, no language conditioning" — no `dataset_composition.pdf` in the deck) →
orchestrator & recovery (`orchestrator_decision_flow.pdf`, given the wide 0.63 column;
the **grasp gate** bullet lives here, not on the decomposition slide) → fine-tuning variants
(**schematic only**: the SmolVLA TikZ from `methodology.tex` ported inline — but
**recoloured**: three orange shades encode the cheapest variant that trains each block
(deep = standard: expert+state proj.; medium = partial adds embedding+LM; pale = full adds
vision encoder), with legend chips; needs `fit,backgrounds,calc` tikzlibraries) →
outcomes across all models
(`orange_outcome_recipes.pdf`; bullets structured as **Subtasking / Auto-data /
Fine-tuning effects** + the 100-seeded line) → grasp bottleneck (bullets left, **both** composition
figures stacked in a wide right column, `failure_modes_variants_subtask.pdf` + `_monotask.pdf`; bullets = grasp
is the bottleneck + orchestrated LIFT advantaged by the privileged gate) → recovery stall
table (`tab:level2_deadline`) → obedience (`grasp_obedience_confusion.pdf`) →
takeaways+future. **Backup** (dark title bar): ACT-vs-SmolVLA baseline
(`baseline_comparison.pdf`), limitations.

Build:
```bash
export PATH="/home/students/texlive/2026/bin/x86_64-linux:$PATH"
cd report/slides && latexmk -pdf -interaction=nonstopmode -halt-on-error slides.tex
```
Both `slides.tex` and the compiled `slides/slides.pdf` are tracked (the PDF is
committed so the deck is viewable without a TeX toolchain; rebuild + recommit it
whenever `slides.tex` changes). The previous PowerPoint pipeline (`presentation.pptx`,
`build_pptx.py`, `assets/`) has been removed in favour of this LaTeX deck.

## Keeping this file current

Update this file **and** `AGENTS.md` in the same commit as any structural change: new plotting scripts, new sections, renamed figures, updated section status. The `.md` and the code must always agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**