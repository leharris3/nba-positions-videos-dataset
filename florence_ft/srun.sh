#!/bin/bash

#SBATCH --partition=a6000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8           
#SBATCH --time=24:00:00

cd /mnt/arc/levlevi/nba-positions-videos-dataset/florence_ft

VENV_NAME="hplr"
PROJ_ROOT=$ENDPOINT/nba-positions-videos-dataset/florence_ft
CONDA_PATH=$ENDPOINT/anaconda3/etc/profile.d/conda.sh
SOUT=$ENDPOINT/nba-positions-videos-dataset/florence_ft/sout/%j.out
JOB_NAME="florence_ft"

config=config.yaml
batch_size=4
epochs=10
lr=1e-6
eval_steps=1000

# cd $PROJ_ROOT
source $CONDA_PATH
conda activate $VENV_NAME

sbatch -o "$SOUT" -J "$JOB_NAME" --wrap="python ft.py \
    --config $config \
    --batch-size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --eval-steps $eval_steps"
