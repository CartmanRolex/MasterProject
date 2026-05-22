# MasterProject

Three-machine robotics research project. See each subdirectory's CLAUDE.md for machine-specific details.

## Subdirectory Map

| Directory | Machine | Docs |
|-----------|---------|------|
| `isaac-inference/` | Desktop | [CLAUDE.md](isaac-inference/CLAUDE.md) |
| `cluster-training/` | Cluster | [CLAUDE.md](cluster-training/CLAUDE.md) |
| `dataset-editor/` | Laptop | [CLAUDE.md](dataset-editor/CLAUDE.md) |
| `leisaac-mods/` | Desktop | [CLAUDE.md](leisaac-mods/CLAUDE.md) |

## Git Policy

**Tracked:** source files, shell scripts, CLAUDE.md files, eval result `.txt` logs (small).
**Gitignored:** `isaac-inference/${data}/` (NvStreamer logs), `isaac-inference/teleop-datasets/` (236 GB HDF5), `isaac-inference/synthetic_datasets/` (local LeRobot recordings), `cluster-training/bash-out/` and `cluster-training/outputs/` (SLURM outputs and checkpoints), `__pycache__/`.

After making repository changes, commit the relevant files and push the branch
to `origin`. Keep unrelated dirty working-tree changes out of the commit.
