"""
1. Either extract frames from videos and treat as image sequence or use a video loader
2. Convert bbox annotations to COCO format compatible with https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py
3. max out batch size and write results when running evaluation script --out flag dump json.
---------------------------------------------------------------------------------------------
Q: where tf is this eval script?
for clip in clips
    results_clip = {}
    for frame in clip:
        if not frame_id: skip
        for bbx in bbxs:
            results_clip[frame][player_id] = model(frame[bbx])
do we need a 2-stage approach for max batch processing?
---------------------------------------------------------------------------------------------
"""

import argparse
import json
import yaml
import logging
import torch.multiprocessing as mp
import cv2
import numpy as np
import codecs, json
import os
import typing
import time
import torch
import concurrent

from typing import List
from glob import glob
from huggingface_hub import hf_hub_download
from easy_ViTPose.vit_utils.inference import pad_image
from easy_ViTPose import VitInference
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_unique_frame_ids(annotations):
    unique_ids = {}
    for index, frame in enumerate(annotations["frames"]):
        frame_id = frame["frame_id"]
        unique_ids[frame_id] = index
    return unique_ids


def process_image(img, model):
    
    # TODO: we are gettting occasional ZeroDivisionError errors
            # using a pretty shitty workaround atm
    try:
        img, _ = pad_image(img, 3 / 4)
        img_input, org_h, org_w = model.pre_img(img)
    except Exception as e:
        logger.error(f"Failed to pad image with shape {img.shape}, error: {e}")
        # use a dummy image
        img = np.zeros((256, 192, 3), dtype=np.uint8)
        img, _ = pad_image(img, 3 / 4)
        img_input, org_h, org_w = model.pre_img(img)
    return img_input, (org_h, org_w)


def process_bbxs(config, model: VitInference, bbxs: List):
    """
    Predict the keypoints for all bounding boxes in a single frame.
    """

    @torch.no_grad()
    def _batch_inference_torch(imgs: List[np.ndarray]) -> List[np.ndarray]:

        og_dims = []
        batch = []
        
        # using threadpool for now
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1024) as executor:
            future_to_img = {executor.submit(process_image, img, model): img for img in imgs}
            for future in concurrent.futures.as_completed(future_to_img):
                img_input, dims = future.result()
                batch.append(img_input)
                og_dims.append(dims)
                
        # stack imgs in batch
        batch = np.concatenate(batch, axis=0)
        logger.debug(f"pre-processing took {time.time() - start} seconds")

        # copy tensor -> cuda
        batch = torch.from_numpy(batch).to(torch.device(model.device))
        
        # forward pass
        # TODO: when using a compiled model, we seem to get "hung up" on previously unseen batch sizes
        # this is not an issue on subsequent batches
        start = time.time()
        heatmaps = model._vit_pose(batch).detach().cpu().numpy()
        logger.debug(f"forward pass took {time.time() - start} seconds")

        # post process results
        start = time.time()
        post_processed_imgs = []
        for i, (org_h, org_w) in enumerate(og_dims):
            heatmap = heatmaps[i, :, :, :]
            heatmap = np.expand_dims(heatmap, axis=0)
            post_processed_imgs.append(model.postprocess(heatmap, org_w, org_h)[0])
        logger.debug(f"post processing imgs took {time.time() - start} seconds")
        
        return post_processed_imgs

    batches = []
    results = []
    
    # TODO: a batch size of 512 only uses ~1/4 GPU memory
    # we can fit multiple clips into mem at once
    batch_size = 1024
    
    # generate batches
    for i in range(0, len(bbxs), batch_size):
        batches.append(bbxs[i : i + batch_size])
        
    for batch in tqdm(batches, desc="generating keypoints for bounding boxes"):
        results.extend(_batch_inference_torch(batch))
    return results


def process_video(config, annotation_path: str, model: VitInference):
    
    # load annotations
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    # get all unqiue frame ids
    unique_frame_ids = get_unique_frame_ids(annotations)
    root_dir = config["nba_ds_root_dir"]
    video_path = root_dir + annotations["video_path"]
    if not os.path.exists(video_path):
        logger.error(f"{video_path} not found")
        return False

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bounding_boxes = []
    bounding_box_map = {}
    
    # optimize these values
    x_pad = 15
    y_pad = 15

    # 1. load all frames from a clip into memory
    # 2. create a list of bbxs for entire clip [bbx1, bbx2, ...]
    # 3. create a mapping {bbx_idx: [frame_id, bbx_index]}
    # 4. process bbxs in batches of 64, get [results1, results2, ...]
    # 5. update annotations file
    # 6. write results
    
    start = time.time()
    frame_idx = 0
    for i, frame_idx in zip(range(frame_count), unique_frame_ids.keys()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_obj = annotations["frames"][i]
        for bbx_frame_obj_idx, bbx in enumerate(frame_obj["bbox"]):
            # crop frame
            x, y, width, height = (
                min(int(bbx["x"]) - x_pad, 0),
                min(int(bbx["y"]) - y_pad, 0),
                int(bbx["width"]) + (2 * x_pad),
                int(bbx["height"]) + (2 * y_pad),
            )
            
            # occasionally we run into imgs of size 0
            # a simple but non-cost efficient fix:
            width += 1
            height += 1
                
            cropped_frame = frame[y : y + height, x : x + width]
            bounding_boxes.append(cropped_frame)
            bbx_idx = len(bounding_boxes) - 1
            bounding_box_map[bbx_idx] = (i, bbx_frame_obj_idx)
            
    end = time.time()
    logger.debug(f"loading frames took {end - start} seconds")
        
    results = process_bbxs(config, model, bounding_boxes)
    for idx, result in enumerate(results):
        frame_idx, bbx_frame_obj_idx = bounding_box_map[idx]
        annotations["frames"][frame_idx]["bbox"][bbx_frame_obj_idx]["keypoints"] = result
        
    # write results
    out_fp = (
        config["results_dir"]
        + "/"
        + annotations["video_path"].replace(".mp4", "_annotation.json")
    )

    # make dirs as needed
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    logger.info(f"writing results to {out_fp}")
    with open(out_fp, "w") as f:
        json.dump(annotations, f, indent=4, cls=NumpyEncoder)
    return True


def main(config):
    # TODO: distribute across gpus
    # q: how much memory are we currently using with bs = 1?

    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)
    logger.info(f"loading model from {model_path}")

    # could yolo size be reduced?
    model = VitInference(
        model_path,
        yolo_path,
        MODEL_SIZE,
        dataset=DATASET,
        yolo_size=320,
        is_video=False,
    )

    annotation_file_paths = glob(config["clips_annotations_dir"] + "/*/*/*.json")
    
    files_processed = 0
    for annotation_file_path in annotation_file_paths:
        result = process_video(config, annotation_file_path, model)
        if not result:
            continue
        files_processed += 1
        if files_processed % 5 == 0:
            break


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
