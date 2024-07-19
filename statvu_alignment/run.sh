#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

# configuration
PYTHON_EXECUTABLE="python3"
CONDA_PATH=/mnt/opr/levlevi/anaconda3/etc/profile.d/conda.sh
SCRIPT_NAME="extract_time_remaining_pipeline.py"
CONFIG_FILE="config.yaml"
VENV_NAME="hplr"

# cd into the project root
source $CONDA_PATH

# activate virtual environment
conda activate $VENV_NAME

$PYTHON_EXECUTABLE $SCRIPT_NAME --config $CONFIG_FILE

# deactivate virtual environment
conda deactivate