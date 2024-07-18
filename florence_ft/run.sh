#!/bin/bash

#SBATCH --partition=a6000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8           
#SBATCH --time=24:00:00

VENV_NAME="hplr"
PROJ_ROOT=/playpen-storage/levlevi/nba-positions-videos-dataset/florence_ft
CONDA_PATH=/playpen-storage/levlevi/anaconda3/etc/profile.d/conda.sh

config=config.yaml
batch_size=4
epochs=10
lr=1e-6
eval_steps=1000

export CUDA_VISIBLE_DEVICES=4,5,6,7
cd $PROJ_ROOT
source $CONDA_PATH
conda activate $VENV_NAME

python ft.py \
    --config $config \
    --batch-size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --eval-steps $eval_steps \