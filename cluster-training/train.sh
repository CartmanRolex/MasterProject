#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --qos debug
#SBATCH --ntasks 1
#SBATCH --time 01:00:00
#SBATCH --cpus-per-task 8
#SBATCH --output=bash-out/train_test.out
#SBATCH --job-name=lerobot_train_test

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nvidia-smi

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=MasterProject2026/pick-orange-mimic-test \
  --dataset.repo_id=leisaac-pick-orange-mimic-v0 \
  --dataset.root=/scratch/izar/gpascual/leisaac-pick-orange-mimic-v0 \
  --batch_size=16 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=outputs/train/pick-orange-mimic-test \
  --job_name=pick-orange-mimic-test \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=8 \