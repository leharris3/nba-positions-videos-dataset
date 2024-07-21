#!/bin/bash
#SBATCH --nodes=megatron.ib
#SBATCH --nodelist=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

VENV_NAME="hplr"
PROJ_ROOT=/mnt/arc/levlevi/nba-positions-videos-dataset/scene-filtering
CONDA_PATH=/mnt/opr/levlevi/anaconda3/etc/profile.d/conda.sh
SOUT=/mnt/arc/levlevi/nba-positions-videos-dataset/scene-filtering/sout/%j.out
JOB_NAME="scene_parse_yolo_ft"

cd $PROJ_ROOT
source $CONDA_PATH
conda activate $VENV_NAME

sbatch -o "$SOUT" -J "$JOB_NAME" --wrap="python3 ft.py"