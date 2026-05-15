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

# Report (laptop)

LaTeX toolchain lives on this laptop. Inference and eval results are
produced on the desktop and synced via git. Do not attempt to run
inference from this machine.

## Report state (May 2026)

Sections in `report/`:
- `introduction.tex` — draft
- `related_work.tex` — draft
- `methodology.tex` — DRAFT, NEEDS REWRITE around the three-mechanism
  framing in the project context above. Currently uses "recovery" as
  an umbrella term; this is the central thing to fix.
- `experiments.tex` — draft
- `results.tex` — partial; only 3-run pilot numbers so far. Full
  100-episode eval pending (see desktop priorities).
- `conclusion.tex` — draft
- `abstract.tex` — draft, write last
- `notes.md` — historical scratch pad; do not edit unless asked.

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