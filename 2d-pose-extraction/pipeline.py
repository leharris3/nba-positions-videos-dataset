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

from glob import glob
from huggingface_hub import hf_hub_download
from easy_ViTPose import VitInference
from tqdm import tqdm


ext = ".pth"
ext_yolo = ".pt"
MODEL_SIZE = "h"
YOLO_SIZE = "s"
DATASET = "wholebody"
MODEL_TYPE = "torch"
YOLO_TYPE = "torch"
REPO_ID = "JunkyByte/easy_ViTPose"
FILENAME = (
    os.path.join(MODEL_TYPE, f"{DATASET}/vitpose-" + MODEL_SIZE + f"-{DATASET}") + ext
)
FILENAME_YOLO = "yolov8/yolov8" + YOLO_SIZE + ext_yolo

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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


def process_bbxs(config, model: VitInference, frame_obj, frame):
    """
    Predict the keypoints for all bounding boxes in a single frame.
    """
    
    updated_bbxs = []
    bbxs = frame_obj["bbox"]

    # optimize these values
    x_pad = 15
    y_pad = 15

    # batch of cropped images to process
    batch = np.array([])
    for bbx in bbxs:
        # crop frame
        x, y, width, height = (
            int(bbx["x"]) - x_pad,
            int(bbx["y"]) - y_pad,
            int(bbx["width"]) + (2 * x_pad),
            int(bbx["height"]) + (2 * y_pad),
        )
        cropped_frame = frame[y : y + height, x : x + width]
        # if any dim of bbx is 0, skip
        if width == 0 or height == 0:
            continue
        batch = np.append(batch, cropped_frame)
    
    results = model.inference(batch)
    print(f"Results: {results}")
    assert False
    
        # try:
        #     results = model.inference(cropped_frame)
        # except Exception as e:
        #     logger.error(f"Error processing frame {bbx['frame_id']}: {e}")
        #     continue
        # bbx["keypoints"] = results
        # updated_bbxs.append(bbx)
        
    frame_obj["bbox"] = updated_bbxs
    return frame_obj


def process_video(config, annotation_path: str, model: VitInference):
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
        
    # 1. get all unqiue frame ids
    unique_frame_ids = get_unique_frame_ids(annotations)
    root_dir = config["nba_ds_root_dir"]
    video_path = root_dir + annotations["video_path"]
    if not os.path.exists(video_path):
        logger.error(f"{video_path} not found")
        return False

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read frames
    frame_idx = 0
    for i, frame_idx in zip(tqdm(range(frame_count)), unique_frame_ids.keys()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_obj = annotations["frames"][i]
        updated_frame_obj = process_bbxs(config, model, frame_obj, frame)
        annotations["frames"][i] = updated_frame_obj
        frame_idx += 1

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
    for annotation_file_path in annotation_file_paths:
        result = process_video(config, annotation_file_path, model)
        if not result:
            continue
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