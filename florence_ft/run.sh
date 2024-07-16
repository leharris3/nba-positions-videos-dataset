#!/bin/bash
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

cd /playpen-storage/levlevi/nba-positions-videos-dataset/florence_ft

VENV_NAME="hplr"
source /playpen-storage/levlevi/anaconda3/etc/profile.d/conda.sh

# activate virtual environment
conda activate $VENV_NAME

# run the training script using mpiexec to manage multiple processes
torchrun --nproc_per_node=1 --run-path finetune_flor.py \
    --annotations_fp "/playpen-storage/levlevi/nba-positions-videos-dataset/statvu_alignment/assets/annotations/annotations.json" \
    --save_every 1 \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --world_size 2 \