#!/bin/bash

DESTINATION_DIR="statvu-raw-data"
DRIVE_FOLDER_LINK="https://drive.google.com/drive/folders/1OkPb160SAhD5S5RXjYN1enHAV82tJlzu"
gdown "$DRIVE_FOLDER_LINK" -O "$DESTINATION_DIR"

echo "StatVU dataset downloaded."

# Search for .7z file in the destination folder
7Z_FILE=$(find "$DESTINATION_DIR" -name "*.7z" -exec basename {} \;)

if [ -n "$7Z_FILE" ]; then
    # Unzip the .7z file
    7z x "$DESTINATION_DIR/$7Z_FILE" -o"$DESTINATION_DIR"

    if [ $? -eq 0 ]; then
        echo "File extracted successfully."
        # Delete the .7z file
        rm "$DESTINATION_DIR/$7Z_FILE"
        echo "File deleted."
    else
        echo "Extraction failed."
    fi
else
    echo "No .7z file found in the destination folder."
fi
