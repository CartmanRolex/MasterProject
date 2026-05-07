# MasterProject

Robotics research for a pick-and-place task: a SO-101 robot arm picks oranges and places them on a plate, using Isaac Sim + LeRobot + SmolVLA / ACT policies.

## Three-Machine Setup

| Machine | Directory | Role |
|---------|-----------|------|
| Desktop | `isaac-inference/` | Policy evaluation and dataset recording via Isaac Sim |
| Laptop | `dataset-editor/` | Manual dataset annotation GUI |
| Cluster (EPFL SCITAS) | `cluster-training/` | SLURM training jobs |

## Repository Structure

```
MasterProject/
├── isaac-inference/    ← Inference scripts, eval results, remote.sh launcher
│   └── results/        ← All eval logs (git-tracked) + plot.py
├── cluster-training/   ← SLURM train/resume scripts
├── dataset-editor/     ← GUI editor + lerobot_editor package + check_scripts
└── leisaac-mods/       ← Git patch and custom modules for LeIsaac
```

Model weights are on HuggingFace Hub under [`MasterProject2026`](https://huggingface.co/MasterProject2026), not in this repo.

See the `CLAUDE.md` in each subdirectory for machine-specific setup and known issues.

## Quick Start

**Evaluation (Desktop)**
```bash
cd isaac-inference
./remote.sh inference_autonomous_orders.py
```

**Training (Cluster)**
```bash
cd cluster-training
mkdir -p bash-out
sbatch train.sh
```

**Dataset Editing (Laptop)**
```bash
cd dataset-editor
python editor.py MasterProject2026/my-dataset
```

