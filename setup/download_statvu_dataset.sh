#!/bin/bash

DESTINATION_DIR="statvu-raw-data"
DRIVE_FOLDER_LINK="https://drive.google.com/drive/folders/11Qr3WjjjvuyXema2sRECvh63RsK9Dx7l?usp=drive_link"
gdown "$DRIVE_FOLDER_LINK" -O "$DESTINATION_DIR"

echo "StatVU dataset downloaded."