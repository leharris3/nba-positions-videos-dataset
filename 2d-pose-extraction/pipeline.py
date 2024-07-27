"""
1. Either extract frames from videos and treat as image sequence or use a video loader
2. Convert bbox annotations to COCO format compatible with https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py
3. max out batch size and write results when running evaluation script --out flag dump json.
"""

import argparse
import concurrent.futures
import json
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import yaml
import logging
import torch.multiprocessing as mp
import cv2
import numpy as np
import json
import os
import time
import torch
import gc
import concurrent

from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
from typing import List, Dict, Optional
from glob import glob
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from easy_ViTPose import VitInference
from utils.image_processing import pre_process_image, post_process_image
from utils.data import NBAClips

# to avoid an error we get when calling torch.compile
torch.set_float32_matmul_precision("high")

EXT = ".pth"
EXT_YOLO = ".pt"
MODEL_SIZE = "h"
YOLO_SIZE = "s"
DATASET = "wholebody"
MODEL_TYPE = "torch"
YOLO_TYPE = "torch"
REPO_ID = "JunkyByte/easy_ViTPose"
FILENAME = (
    os.path.join(MODEL_TYPE, f"{DATASET}/vitpose-" + MODEL_SIZE + f"-{DATASET}") + EXT
)
FILENAME_YOLO = "yolov8/yolov8" + YOLO_SIZE + EXT_YOLO

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ViTPoseCustom:
    """
    Custom model class for a easy_vit keypoint detection model.
    """

    def __init__(self, model: torch.nn.Module, device: int):
        self.model = model.to(device).eval()
        self.model = torch.compile(self.model).half()


def worker(device: str, config: Dict, annotation_fps: List[str]) -> None:
    # process we run multiple times on the same gpu
    # load model in each process
    infer_model_obj = VitInference(
        config["model_path"],
        config["yolo_path"],
        MODEL_SIZE,
        dataset=DATASET,
        yolo_size=320,
        is_video=False,
        device="cpu",
    )

    # create a custom model object
    model: ViTPoseCustom = ViTPoseCustom(infer_model_obj._vit_pose, device)
    del infer_model_obj

    logger.info(f"Creating dataset")
    dataset = NBAClips(config, annotation_fps, device)

    logger.info(f"Creating dataloader")
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=False,
    )

    # lil test for now
    with torch.no_grad():
        for i, (
            batch,
            curr_annotation_fp_idx,
            curr_frame_idx,
            curr_rel_bbx_idx,
            og_h,
            og_w,
        ) in enumerate(dataloader):
            print(curr_annotation_fp_idx)
            start = time.time()

            # copy -> gpu
            batch = batch.to(device)
            # forward pass
            _ = model.model(batch)

            # logging
            end = time.time()
            logger.info(f"batch {i} took {end - start} seconds")

            # cleanup
            del batch
            gc.collect()

            # TODO: post-processing


def main(config):

    # initally load model
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)
    logger.info(f"loading model from {model_path}")

    config["model_path"] = model_path
    config["yolo_path"] = yolo_path

    # all annotation paths
    annotation_file_paths = glob(config["clips_annotations_dir"] + "/*/*/*.json")
    logger.info(f"{len(annotation_file_paths)} total files to process")

    # fairly sure num_proc=8 per gpus best
    num_gpus = torch.cuda.device_count()
    num_gpus = 1

    # split annotation file paths among workerss
    file_chunks = [annotation_file_paths[i::num_gpus] for i in range(num_gpus)]

    # create pool of workers
    # don't need a manager atm
    manager = mp.Manager()
    for gpu_id in range(num_gpus):
        mp.spawn(
            worker,
            args=(config, file_chunks[gpu_id]),
            nprocs=1,
            join=True,
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config", type=str, required=True, help="path to yaml config file"
    )
    args = args.parse_args()
    # load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
