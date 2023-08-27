#!/bin/bash

DESTINATION_DIR="unprocess-videos"
DRIVE_FOLDER_LINK="https://drive.google.com/drive/folders/1ODZDZsWWsXpTmJnxU016spv6KJGQQkF0"
gdown "$DRIVE_FOLDER_LINK" -O "$DESTINATION_DIR"

echo "Estimated size of download: $ESTIMATED_SIZE"
