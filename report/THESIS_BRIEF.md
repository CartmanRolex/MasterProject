# Master's Thesis — Official Brief (EPFL / CREATE Lab)

Source: Notion "Thesis Student Info"
<https://boulder-gazelle-c8c.notion.site/Thesis-Student-Info-3411a8cb4b4948edb8e2058f7a728fd1>
(JS-rendered page; extracted 2026-06-25. This file is the canonical local copy so the
requirements don't get lost.)

---

## Report specifications (HARD CONSTRAINTS)

- **Length: max 20 pages.** "Quality over quantity please." This is a firm ceiling.
- Thesis-writing advice deck referenced from the brief (see report/CLAUDE.md for the link).

## Prescribed report structure

> Note: the brief folds the **state-of-the-art / related-work review into the
> Introduction** — it does *not* prescribe a standalone "Related Work" section.
> The Methods step also asks for a brief "nearest 3-4 papers" review.

**Introduction**
- What is the problem.
- What is the goal — *"include review or related state-of-the-art to show what has
  already been explored and hence what is the goal of this work."*
- What is the hypothesis you are proposing/testing.
- What do you show or demonstrate in your work.
- What are the key contributions.
- *Optional* sub-section: Problem Statement.

**Methods**
- What methods have you developed / what experimental processes undertaken
  (model formation, simulation formulation, fabrication methods).
- May include figures relating to the methods.

**Results**
- Experimental results.
- "Try and avoid time series data unless absolutely necessary."
- "The experiments should map back to your hypothesis in the introduction."

**Discussion & Conclusion**
- What has been learnt/contributed and how it answers the hypothesis.
- What improvements / further extensions could be made.

**References**

**Supplementary Information**

## Project-plan content the brief expects (Weeks 1-2 — for reference)

**Goal** — main objective; what problem; what capability/method/system should exist by the end.
**Methodology** — main steps; **brief review of related work (nearest 3-4 papers to what
you propose)**; novel approach / hypothesis being tested; planned implementation
(hardware/software); experiments to evaluate.
**Expected outcomes** — working prototype/system/algorithm; experimental evaluation; insights/guidelines.
**Timeline & key milestones** — approximate dates for experiments / thesis draft / etc.

## Deadlines & milestones (process)

- Outline & structure of the report: **≥4 weeks before deadline**.
- Report draft: **1-2 weeks before deadline**.
- Mid-way presentation: not assessed; for feedback from a wider group.
- **Final report and final presentation (with external examiner) affect the grade.**
- Final report due per IS-Academia; consider relevant conference/paper deadlines.

## General advice captured

- Quality over quantity; a report is a structured reflection, not a list of things done.
- Experiments must map back to the hypothesis stated in the Introduction.
- Avoid time-series plots unless unavoidable; every figure needs labels, units, a message.

---

## How this maps to *our* report (Enabling Recovery in VLA-Based Robotic Manipulation)

- Current sections in `report/sections/`: abstract, introduction, methodology,
  experiments, results (= Discussion & Conclusion), plus **empty** related_work.tex
  and conclusion.tex that are **not** `\input` in `main.tex`.
- The literature review now lives in `sections/related_work.tex` (5 thematic clusters,
  21 citations) with `references.bib` expanded accordingly; both are wired into
  `main.tex` (`\bibliography`, `ieeetr` style).
</content>
