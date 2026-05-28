# MasterProject

Three-machine robotics research project. See each subdirectory's AGENTS.md for machine-specific details.

## Subdirectory Map

| Directory | Machine | Docs |
|-----------|---------|------|
| `isaac-inference/` | Desktop | [AGENTS.md](isaac-inference/AGENTS.md) |
| `cluster-training/` | Cluster | [AGENTS.md](cluster-training/AGENTS.md) |
| `dataset-editor/` | Laptop | [AGENTS.md](dataset-editor/AGENTS.md) |
| `leisaac-mods/` | Desktop | [AGENTS.md](leisaac-mods/AGENTS.md) |
| `report/` | Laptop | [AGENTS.md](report/AGENTS.md) |

## Git Policy

**Tracked:** source files, shell scripts, CLAUDE.md and AGENTS.md files, eval result `.txt` logs (small).
**Gitignored:** `isaac-inference/${data}/` (NvStreamer logs), `isaac-inference/teleop-datasets/` (236 GB HDF5), `isaac-inference/synthetic_datasets/` (local LeRobot recordings), `cluster-training/bash-out/` and `cluster-training/outputs/` (SLURM outputs and checkpoints), `__pycache__/`.

After making repository changes, commit the relevant files and push the branch
to `origin`. Keep unrelated dirty working-tree changes out of the commit.
