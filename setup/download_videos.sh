#!/bin/bash

DESTINATION_DIR="unprocess-videos"
DRIVE_FOLDER_LINK="https://drive.google.com/drive/folders/1ODZDZsWWsXpTmJnxU016spv6KJGQQkF0"
gdown --folder "$DRIVE_FOLDER_LINK" -O "$DESTINATION_DIR"
