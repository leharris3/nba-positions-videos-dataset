#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

cd /playpen-storage/levlevi/nba-positions-videos-dataset/florence_ft

batch_size=2

VENV_NAME="hplr"
source /playpen-storage/levlevi/anaconda3/etc/profile.d/conda.sh

# activate virtual environment
conda activate $VENV_NAME

python ft.py \
    --config config.yaml \
    --batch-size $batch_size \
    --epochs 10 \
    --lr 1e-6 \
    --eval-steps 1000 \