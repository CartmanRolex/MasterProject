# Cluster Training

SLURM job scripts for training LeRobot policies on the EPFL SCITAS cluster. All scripts assume the `lerobot` conda environment.

## Scripts

| File | GPUs | QOS | Purpose |
|------|------|-----|---------|
| `train.sh` | 1 | debug (1 h) | Single-GPU training run |
| `train_xl.sh` | 4 (2 nodes × 2) | normal (6 h) | Multi-GPU distributed training via `accelerate launch` |
| `resume.sh` | 1 | normal (2 h) | Resume single-GPU training from a checkpoint |
| `resume_xl.sh` | 4 (2 nodes × 2) | normal (6 h) | Resume multi-GPU training from a checkpoint |

## Usage

```bash
mkdir -p bash-out   # required before first sbatch
sbatch train.sh
sbatch train_xl.sh
```

## Output Paths (gitignored)

- SLURM stdout logs → `bash-out/` (create manually before submitting)
- Model checkpoints → `outputs/train/<job_name>/` (large, not tracked)

## Dataset on Cluster

Datasets are stored on scratch: `/scratch/izar/gpascual/<dataset_name>/`. Pass via `--dataset.root=...`.

## Conda

The SLURM scripts activate conda automatically. To activate manually:
```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot
```
