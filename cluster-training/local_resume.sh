#!/usr/bin/env bash
# Resume a local training run from a checkpoint's train_config.json (single GPU).
#
# Usage:
#   cluster-training/local_resume.sh <.../checkpoints/NNNNNN/pretrained_model/train_config.json>
# Example:
#   cluster-training/local_resume.sh ~/Gal/outputs/train/Gal-merged-tailed-auto/checkpoints/080000/pretrained_model/train_config.json
set -euo pipefail

CONFIG="${1:?usage: local_resume.sh <path/to/train_config.json>}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
lerobot-train --config_path="$CONFIG" --resume=true
