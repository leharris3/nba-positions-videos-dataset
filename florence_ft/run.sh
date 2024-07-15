#!/bin/bash

# set strict mode
set -euo pipefail

# configuration
PYTHON_EXECUTABLE="python3"
SCRIPT_NAME="finetune_flor.py"
CONFIG_FILE="config.yaml"

$PYTHON_EXECUTABLE $SCRIPT_NAME --config $CONFIG_FILE