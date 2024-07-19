#!/bin/bash
#SBATCH --nodelist=megatron.ib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

VENV_NAME="hplr"
PROJ_ROOT=$ENDPOINT/nba-positions-videos-dataset/florence_ft
CONDA_PATH=/mnt/opr/levlevi/anaconda3/etc/profile.d/conda.sh
SOUT=$ENDPOINT/nba-positions-videos-dataset/florence_ft/sout/%j.out
JOB_NAME="florence_ft"

config=config.yaml
batch_size=1
epochs=10
lr=1e-6
eval_steps=1000

cd $PROJ_ROOT
source $CONDA_PATH
conda activate $VENV_NAME

sbatch -o "$SOUT" -J "$JOB_NAME" --wrap="python3 ft.py \
    --config $config \
    --batch-size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --eval-steps $eval_steps"
