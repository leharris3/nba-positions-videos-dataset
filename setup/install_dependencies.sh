#!/bin/bash

# MARK: installs all dependencies of this project for linux-based system.
pip install deepsport-utilities
pip install torch torchvision
pip install tqdm yacs
pip install pytorch-ignite
pip install pytesseract
pip install object-detection
pip install py7zr
pip install Pillow
pip install gdown
apt-get update
apt-get install tesseract-ocr
apt-get install libtesseract-dev