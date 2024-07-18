#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# set strict mode
cd /mnt/arc/levlevi/nba-positions-videos-dataset/statvu_alignment
set -euo pipefail

# configuration
PYTHON_EXECUTABLE="python3"
SCRIPT_NAME="extract_time_remaining_pipeline.py"
CONFIG_FILE="config.yaml"
VENV_NAME="hplr"

# cd into the project root
source /playpen-storage/levlevi/anaconda3/etc/profile.d/conda.sh

# activate virtual environment
conda activate $VENV_NAME

$PYTHON_EXECUTABLE $SCRIPT_NAME --config $CONFIG_FILE

# deactivate virtual environment
conda deactivate