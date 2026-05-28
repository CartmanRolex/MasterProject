# MasterProject

Three-machine robotics research project. See each subdirectory's CLAUDE.md for machine-specific details.

## Subdirectory Map

| Directory | Machine | Docs |
|-----------|---------|------|
| `isaac-inference/` | Desktop | [CLAUDE.md](isaac-inference/CLAUDE.md) |
| `cluster-training/` | Cluster | [CLAUDE.md](cluster-training/CLAUDE.md) |
| `dataset-editor/` | Laptop | [CLAUDE.md](dataset-editor/CLAUDE.md) |
| `leisaac-mods/` | Desktop | [CLAUDE.md](leisaac-mods/CLAUDE.md) |
| `report/` | Laptop | [CLAUDE.md](report/CLAUDE.md) |

## Git Policy

**Tracked:** source files, shell scripts, CLAUDE.md and AGENTS.md files, eval result `.txt` logs (small).
**Gitignored:** `isaac-inference/${data}/` (NvStreamer logs), `isaac-inference/teleop-datasets/` (236 GB HDF5), `isaac-inference/synthetic_datasets/` (local LeRobot recordings), `cluster-training/bash-out/` and `cluster-training/outputs/` (SLURM outputs and checkpoints), `__pycache__/`.

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
