# MasterProject Agent Instructions

This repository uses `CLAUDE.md` files for detailed project context. Before
editing a subdirectory, read the root `CLAUDE.md` and the nearest
subdirectory-specific `CLAUDE.md`.

## Git Workflow

- After making repository changes, commit the relevant files and push the
  branch to `origin`.
- Keep unrelated dirty working-tree changes out of the commit.
- Do not commit ignored large artifacts such as datasets, logs, checkpoints
  from training output directories, or cache files.
- Small evaluation summaries in `isaac-inference/results/` are intended to be
  tracked.
