#!/usr/bin/env bash
# Local single-GPU SmolVLA training on this desktop's RTX 5090.
# Replaces the SLURM scripts (SCITAS dropped). Datasets resolve from the local
# HF cache by repo_id (MasterProject2026/<name>); no --dataset.root needed.
#
# Only ONE GPU-heavy job at a time on the shared 5090 (this script warns if the
# GPU is already busy).
#
# Usage:
#   cluster-training/local_train.sh <dataset_repo_id> <job_name> [steps] [batch_size]
# Example:
#   cluster-training/local_train.sh MasterProject2026/Gal-merged-tailed-auto Gal-merged-tailed-auto
#
# Env overrides: OUTPUT_ROOT (default ~/Gal/outputs/train), WANDB_ENABLE (default false).
set -euo pipefail

DATASET="${1:?usage: local_train.sh <dataset_repo_id> <job_name> [steps] [batch_size]}"
JOB="${2:?usage: local_train.sh <dataset_repo_id> <job_name> [steps] [batch_size]}"
STEPS="${3:-20000}"
BATCH="${4:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/Gal/outputs/train}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# refuse-ish to start if the single shared GPU is already busy
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)"
echo "GPU memory in use: ${used} MiB"
if [ "${used:-0}" -gt 2000 ]; then
  echo "WARNING: GPU already using ${used} MiB — another job may hold it."
  echo "Ctrl-C to abort, or wait 8s to continue..."
  sleep 8
fi

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="MasterProject2026/$JOB" \
  --dataset.repo_id="$DATASET" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --output_dir="$OUTPUT_ROOT/$JOB" \
  --job_name="$JOB" \
  --policy.device=cuda \
  --wandb.enable="$WANDB_ENABLE" \
  --num_workers=8
