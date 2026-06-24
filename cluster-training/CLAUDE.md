# Training

SmolVLA / LeRobot policy training. **Now runs locally on the basement desktop's
RTX 5090** — the SCITAS cluster is no longer used (it's slower than this box). The
SLURM scripts are kept only as reference for the old cluster setup.

All scripts assume the `lerobot` conda environment. (The directory is still named
`cluster-training/` for git history; treat it as "training".)

## Local scripts (use these)

| File | Purpose |
|------|---------|
| `local_train.sh` | `<dataset_repo_id> <job_name> [steps] [batch_size]` — single-GPU `lerobot-train` directly on the 5090. Datasets resolve from the local HF cache by repo_id (`MasterProject2026/<name>`). Warns if the GPU is already busy. Env: `OUTPUT_ROOT` (default `~/Gal/outputs/train`), `WANDB_ENABLE` (default `false`). |
| `local_resume.sh` | `<train_config.json>` — resume a local run from a checkpoint's `pretrained_model/train_config.json`. |

```bash
cluster-training/local_train.sh MasterProject2026/Gal-merged-tailed-auto Gal-merged-tailed-auto
# watch it / get pinged: tail -f ~/.claude/cc-waiting.log  (or /loop a status check)
```

Only **one** GPU-heavy job at a time (single 5090 shared with Isaac Sim inference and
teleop). Checkpoints → `~/Gal/outputs/train/<job_name>/` (large, gitignored).

## Legacy SLURM scripts (reference only — SCITAS dropped)

| File | GPUs | Purpose |
|------|------|---------|
| `train.sh` | 1 | Single-GPU SLURM run (`sbatch`). |
| `train_xl.sh` | 4 (2 nodes × 2) | Multi-GPU distributed via `accelerate launch`. |
| `resume.sh` | 1 | Resume single-GPU from a checkpoint. |
| `resume_xl.sh` | 4 | Resume multi-GPU. |

These reference cluster paths (`/scratch/izar/gpascual/...`, `/home/gpascual/...`) and
SLURM QOS that don't exist on this machine. Do not run them here.

## Conda

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot
```

## Keeping this file current

Update this file **and** `AGENTS.md` in the same commit as any structural change: new
training scripts, changed configs, renamed files. The `.md` and the code must always
agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**
