#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=mirage.ib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=32

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