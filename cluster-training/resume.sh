#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --qos normal
#SBATCH --ntasks 1
#SBATCH --time 02:00:00
#SBATCH --cpus-per-task 6
#SBATCH --output=bash-out/resume.out
#SBATCH --job-name=lerobot_train

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --config_path=/home/gpascual/MasterProject/outputs/train/ACT-pick-orange/checkpoints/080000/pretrained_model/train_config.json \
  --resume=true \
