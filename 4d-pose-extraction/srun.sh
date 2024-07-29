#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodelist=bumblebee.ib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=50:00:00

VENV_NAME="2dpose"
PROJ_ROOT=$ENDPOINT/nba-positions-videos-dataset/2d-pose-extraction
CONDA_PATH=/mnt/opr/levlevi/anaconda3/etc/profile.d/conda.sh
SOUT=$PROJ_ROOT/out/%j.out
JOB_NAME="2d-pose-extraction"

config=config.yaml

cd $PROJ_ROOT
source $CONDA_PATH
conda activate $VENV_NAME

/mnt/opr/levlevi/anaconda3/envs/2dpose/bin/python pipeline.py --config $config