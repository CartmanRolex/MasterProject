#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=bash-out/train_subtask.out
#SBATCH --job-name=lerobot_train_subtask

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nvidia-smi

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun bash -c "
  accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=2 \
    --machine_rank=\$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    \$(which lerobot-train) \
    --policy.path=lerobot/smolvla_base \
    --policy.repo_id=MasterProject2026/pick-orange-mimic-subtasked3103 \
    --dataset.repo_id=pick-orange-mimic-subtasked \
    --dataset.root=/scratch/izar/gpascual/pick-orange-mimic-subtasked \
    --batch_size=64 \
    --steps=5000 \
    --save_freq=1000 \
    --output_dir=outputs/train/pick-orange-mimic-subtasked3103 \
    --job_name=pick-orange-mimic-subtasked \
    --wandb.enable=true \
    --num_workers=8 \
    --policy.optimizer_lr=0.0004 \
    --policy.device=cuda
"