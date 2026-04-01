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
    --config_path=/home/gpascual/MasterProject/outputs/train/pick-orange-mimic-subtasked3103/checkpoints/004000/pretrained_model/train_config.json \
    --resume=true \
"