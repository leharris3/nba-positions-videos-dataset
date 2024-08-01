import os
import cv2
import hydra
import warnings
import numpy as np
import json
import yaml
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from glob import glob
from utils.data import NBAClips

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.utils import get_pylogger

from PHALP_MOD import PHALP

warnings.filterwarnings("ignore")
log = get_pylogger(__name__)

from typing import List, Dict



def worker(rank, config: Dict, hydra_config: DictConfig, model, file_paths: List[str]):
    """
    One torch worker.
    """
    
    # TODO: we will to match the input format of the VitDetDataset to use model forward oob
    # 1. get dataset and dataloader
        # dataloader
            # (bbx_tensors, annotation_fp, frame_idx, bbx_idx)
    # 2. copy model to gpu
    # 3. for batch in dataloader: predict -> post-process -> write
    
    phalp_tracker = PHALP(hydra_config)
    dataset = NBAClips(config, file_paths, rank)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    # # TODO: skeleton
    # for batch in dataloader:
    #     phalp_tracker.predict(batch)
    pass


def main(config: Dict, hydra_config: DictConfig):
    """
    Main process.
    """
    
    src_dir = config["clips_annotations_dir"]
    dst_dir = config["results_dir"]
    annotation_file_paths = glob(os.path.join(src_dir, "*/*/*.json"))
    processed_annotations = set(
        [
            fp.replace(dst_dir, src_dir)
            for fp in glob(os.path.join(dst_dir, "*/*/*.json"))
        ]
    )
    annotation_file_paths = list(set(annotation_file_paths) - processed_annotations)
    log.info(f"{len(annotation_file_paths)} total remaining files to process")

    num_gpus = config["num_gpus"]

    # split annotation file paths among workerss
    file_chunks = [annotation_file_paths[i::num_gpus] for i in range(num_gpus)]

    # create pool of workers
    mp.spawn(
        worker,
        args=(config, hydra_config, file_chunks),
        nprocs=config["num_gpus"],
        join=True,
    )

if __name__ == '__main__':
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    with open(config["hydra_config"], "r") as f:
        hydra_config = DictConfig(json.load(f))

    main(config, hydra_config)