# MasterProject

Robotics research project: SmolVLA fine-tuning for SO-101 pick-and-place in Isaac
Sim, with a scripted subtask orchestrator (GRASP / LIFT / PLACE). See each
subdirectory's CLAUDE.md for details.

**Single machine.** Everything now runs on the basement desktop (RTX 5090).
Historically the work spanned four machines (Dell laptop; the room desktop where most
datasets were created; this basement desktop; the SCITAS cluster). They are no longer
used — **if a source file or dataset is missing here, it is on the room desktop or the
Dell laptop** and should be copied over. The 236 GB raw `teleop-datasets/` HDF5 in
particular still lives on the room desktop; the trainable LeRobot datasets are in this
machine's HF cache and on the Hub.

See **[WORKFLOW.md](WORKFLOW.md)** for how to run Claude here (parallel subagents,
phone control via `/rc`, unattended runs, notifications) and **[WORKLOG.md](WORKLOG.md)**
to coordinate active sessions.

## Subdirectory Map

| Directory | Purpose | Docs |
|-----------|---------|------|
| `isaac-inference/` | Policy evaluation, orchestrator, dataset pipeline (Isaac Sim) | [CLAUDE.md](isaac-inference/CLAUDE.md) |
| `cluster-training/` | SmolVLA training — now **local** on the 5090 (SLURM scripts kept as reference) | [CLAUDE.md](cluster-training/CLAUDE.md) |
| `dataset-editor/` | Manual dataset annotation GUI | [CLAUDE.md](dataset-editor/CLAUDE.md) |
| `leisaac-mods/` | Custom LeIsaac/Isaac modules (teleop devices, Quest3) | [CLAUDE.md](leisaac-mods/CLAUDE.md) |
| `report/` | Thesis (LaTeX + figure/analysis scripts) | [CLAUDE.md](report/CLAUDE.md) |
| `tooling/` | Claude Code workflow scripts (worktrees, unattended, notifications) | [CLAUDE.md](tooling/CLAUDE.md) |
| `.claude/agents/` | Reusable subagents (report-plotter, eval-analyzer, code-refactor) | — |

## Git Policy

**Tracked:** source files, shell scripts, CLAUDE.md and AGENTS.md files, eval result `.txt` logs (small), `.claude/agents/` and `.claude/settings.json` (shared workflow config).
**Gitignored:** `isaac-inference/${data}/` (NvStreamer logs), `isaac-inference/teleop-datasets/` (236 GB HDF5), `isaac-inference/synthetic_datasets/` (local LeRobot recordings), `cluster-training/bash-out/` and `cluster-training/outputs/` (training outputs/checkpoints), `.claude/` except the two paths above (so `settings.local.json` stays local), `__pycache__/`.

After making repository changes, commit the relevant files and push the branch
to `origin`. Keep unrelated dirty working-tree changes out of the commit.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**

## Documentation maintenance

These rules apply to every agent and every contributor:

- **Always commit and push** after making changes. Never leave the working tree dirty.
- **Keep the .md current.** Whenever you add, remove, rename, or significantly change a file or feature, update the CLAUDE.md (and its AGENTS.md mirror) in the same directory **in the same commit**. The `.md` and the code must always agree.
- **New file added?** Add a row to the relevant table before closing the commit.
- **File deleted or renamed?** Remove or update the corresponding entry.
- **New directory or structural change?** Update the root CLAUDE.md subdirectory map and root AGENTS.md as well.
- CLAUDE.md and AGENTS.md are always identical in a given directory — if you edit one, edit the other.
