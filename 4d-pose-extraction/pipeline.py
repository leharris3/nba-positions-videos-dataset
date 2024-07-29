from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from typing import List, Dict

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full


def worker(rank, config: Dict, model, file_paths: List[str]):
    """
    One torch worker.
    """
    
    # TODO: we will to match the input format of the VitDetDataset to use model forward oob
    # 1. get dataset and dataloader
        # dataloader
            # (bbx_tensors, annotation_fp, frame_idx, bbx_idx)
    # 2. copy model to gpu
    # 3. for batch in dataloader: predict -> post-process -> write
    pass


def main(config: Dict):
    """
    Main process.
    """
    
    # 1. get all fps
    # 2. get one copy of vitdet in shared mem
    pass

if __name__ == '__main__':
    
    # parse args and run
    config = None
    main(config)