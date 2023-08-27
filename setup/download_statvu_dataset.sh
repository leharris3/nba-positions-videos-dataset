#!/bin/bash

DESTINATION_DIR="statvu-raw-data"
DESTINATION_FILE="statvu-raw-data/2016.NBA.Raw.SportVU.Raw.Data.7z"
DRIVE_FOLDER_LINK="https://drive.google.com/drive/folders/1OkPb160SAhD5S5RXjYN1enHAV82tJlzu"
gdown --folder "$DRIVE_FOLDER_LINK" -O "$DESTINATION_DIR"

echo "StatVU dataset downloaded."
7z x "$DESTINATION_FILE" "-o$DESTINATION_DIR"
rm "$DESTINATION_FILE"