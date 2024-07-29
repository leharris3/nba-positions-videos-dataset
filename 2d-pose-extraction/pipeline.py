"""
1. Either extract frames from videos and treat as image sequence or use a video loader
2. Convert bbox annotations to COCO format compatible with https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py
3. max out batch size and write results when running evaluation script --out flag dump json.
"""

import argparse
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import yaml
import logging
import torch.multiprocessing as mp
import numpy as np
import os
import time
import torch
import gc

from torch.utils.data import DataLoader
from multiprocessing import Pool
from typing import List, Dict, Optional
from glob import glob
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from easy_ViTPose import VitInference
from utils.data import NBAClips
from utils.model import ViTPoseCustom
from utils.post_processing import (
    post_process_results,
    update_results,
)
from utils.results import update_results

# to avoid an error we get when calling torch.compile
torch.set_float32_matmul_precision("high")
torch.jit.enable_onednn_fusion(True)
# run a quick benchmark to find the best backend for convolutions
torch.backends.cudnn.benchmark = True


EXT = ".pth"
EXT_YOLO = ".pt"
MODEL_SIZE = "b"
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


@torch.inference_mode()
def worker(
    device: str, config: Dict, model_loader: ViTPoseCustom, annotation_fps: List
) -> None:
    """
    Single worker process for a GPU.
    Represents one instance of a dataset a model.
    PyTorch best practices for inference workloads: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """

    model = model_loader.get_model(device)
    annotation_fps = annotation_fps[int(device)]

    # create dataset
    logger.info(f"Creating dataset")
    dataset = NBAClips(config, annotation_fps, device)

    # create dataloader
    # pinning data to gpu is mainly important in training workloads
    logger.info(f"Creating dataloader")
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=False,
    )

    # main infer loop
    # torch.no_grad() saves lots and lots of mem
    for i, (
        batch,
        curr_annotation_fp_idx,
        curr_frame_idx,
        curr_rel_bbx_idx,
        og_h,
        og_w,
    ) in enumerate(dataloader):

        # copy tesnor cpu -> gpu
        start = time.time()
        batch = batch.to(device)

        # forward pass
        # result = (B, H, W, 3)
        # convert to from fp16 -> f32
        forward_start = time.time()
        # TODO: we expect that heatmaps is going to be an extreamly large tensor
        # faster to keep tensor on GPU and use pytorch ops for post-processing, then copy to cpu later
        if config["use_half_precision"] == "True":
            logger.debug("Casting heatmaps from fp16 -> f32")
            heatmaps = model(batch).float() # .detach().cpu().numpy()
        else:
            heatmaps = model(batch) # .detach().cpu().numpy()
        logger.debug(f"forward pass took {time.time() - forward_start} seconds")

        # post process all results
        logger.debug(f"post-processing batch {i}")
        start_post = time.time()

        # TODO: optimize (7s+ per batch)
        try:
            results = post_process_results(heatmaps, og_w, og_h, device=device)
        except Exception as e:
            logger.error(
                f"Error post-processing batch: {heatmaps.shape}, {og_w}, {og_h}"
            )
            logger.error(e)
            continue
        
        logger.debug(f"post-processing took {time.time() - start_post} seconds")

        # remove errored results
        results = [res for res in results if res is not None]
        if len(results) == 0:
            logger.info(
                "-------------------------------------------------------------------------------------------"
            )
            logger.info(
                f"No valid results in batch for device: {device}, assuming we are done processing all files!"
            )
            logger.info(
                "-------------------------------------------------------------------------------------------"
            )
            del batch
            gc.collect()
            return

        # write results to out
        # TODO: optimize (~5-7s per batch)
        write_start = time.time()
        try:
            update_results(
                config,
                results,
                annotation_fps,
                curr_annotation_fp_idx.tolist(),
                curr_frame_idx.tolist(),
                curr_rel_bbx_idx.tolist(),
            )
        except Exception as e:
            logger.error(f"Error writing results: {heatmaps.shape}, {og_w}, {og_h}")
            logger.error(e)
            continue

        logger.debug(f"writing results took {time.time() - write_start} seconds")
        end = time.time()
        logger.debug(f"batch {i} took {end - start} seconds")

        # cleanup
        del batch
        gc.collect()


def main(config):

    # initally load model
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)

    # easy ViTPose inference onj
    infer_model_obj = VitInference(
        model_path,
        yolo_path,
        MODEL_SIZE,
        dataset=DATASET,
        yolo_size=320,
        is_video=False,
        device="cpu",
    )
    
    # create a custom model object
    model_loader: ViTPoseCustom = ViTPoseCustom(
        config=config, model=infer_model_obj._vit_pose
    )

    logger.info(f"loading model from {model_path}")
    config["model_path"] = model_path
    config["yolo_path"] = yolo_path

    # get all remaining annotation fps to process
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
    logger.info(f"{len(annotation_file_paths)} total remaining files to process")

    num_gpus = config["num_gpus"]

    # split annotation file paths among workerss
    file_chunks = [annotation_file_paths[i::num_gpus] for i in range(num_gpus)]

    # create pool of workers
    mp.spawn(
        worker,
        args=(config, model_loader, file_chunks),
        nprocs=config["num_gpus"],
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
