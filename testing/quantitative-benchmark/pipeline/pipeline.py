import concurrent.futures
import cv2
import torch
import os
import concurrent
import logging
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image
from transformers import logging

from utils.constants import QUARTER_KEY, TIME_REMAINING_KEY, PAD, BREAK, CONF_THRESH
from ocr.tr_ocr import extract_time_remaining_from_images_tr
from ocr.paddle import extract_time_remaining_from_image_paddle
from ocr.minicpm import extract_time_remaining_from_images_minicpm
from ocr.helpers import convert_time_to_float
from ocr.models import TrOCRModel, YOLOModel, PaddleModel, MiniCPMModel


logging.set_verbosity_error()

MAX_GPUS = 8
ROI_STEP = 5
TIME_REMAINING_STEP = 30

ROI_MODELS = {}
MODELS = {}


def process_dir(dir_path: str, data_out_path=None, viz_out_path=None):
    """
    Extract all timestamps in a directory,
    return timestamps as dict.
    """

    assert os.path.isdir(dir_path), f"Error: bad path to video directory: {dir_path}"

    valid_formats = ["avi", "mp4"]
    vids = os.listdir(dir_path)
    for vid in vids:
        extension = vid.split(".")[1]
        if extension not in valid_formats:
            vids.remove(vid)

    timestamps = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_GPUS) as executor:
        with tqdm(total=len(vids), desc="Processing Videos") as pbar:
            while len(vids) > 0:
                processes = []
                video_paths = []
                for device in range(MAX_GPUS):
                    if len(vids) == 0:
                        break
                    video_path = os.path.join(dir_path, vids[0])

                    # data_path = os.path.join(
                    #     data_out_path,
                    #     vids[0].replace(".mp4", ".json").replace(".avi", ".json"),
                    # )

                    process = executor.submit(
                        extract_timestamps_from_video,
                        video_path,
                        # data_path,
                        device=device,
                    )
                    processes.append(process)
                    video_paths.append(video_path)
                    vids.remove(vids[0])
                for process, video_path in zip(
                    concurrent.futures.as_completed(processes), video_paths
                ):
                    vp, ts = process.result()
                    timestamps[vp] = ts
                    pbar.update(1)

    return timestamps


def extract_timestamps_from_video(video_path: str, device: int = 0):
    """
    Given a path to a basketball broadcast video,
    returns a timestamps dict.
    """

    assert os.path.exists(video_path)

    # print(f"Extracting timestamps for video at {video_path} \n")
    tr_x1, tr_y1, tr_x2, tr_y2 = None, None, None, None

    # create YOLO model
    if str(device) not in ROI_MODELS:
        model = YOLOModel(device=device)
        ROI_MODELS[str(device)] = model
    yolo_model = ROI_MODELS[str(device)]

    time_remaining_roi = extract_roi_from_video(video_path, yolo_model, device=device)
    if time_remaining_roi is not None:
        tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = {}
    quarter = video_path[-5]  # period_x.mp4
    step = TIME_REMAINING_STEP

    # create OCR model
    if str(device) not in MODELS:
        model = MiniCPMModel(device=device)
        MODELS[str(device)] = model
    model = MODELS[str(device)]

    for frame_index in range(frames_cnt):
        ret, frame = cap.read()
        if not ret:
            break
        time_remaining_img = None
        time_remaining = None
        if frame_index % step == 0:
            if time_remaining_roi is not None:
                assert tr_x1 and tr_y1 and tr_x2 and tr_y2
                time_remaining_img = frame[
                    tr_y1 - PAD : tr_y2 + 2 * PAD, tr_x1 - PAD : tr_x2 + 2 * PAD
                ]
            if time_remaining_img is not None:
                time_remaining = extract_time_remaining_from_image_paddle(
                    Image.fromarray(time_remaining_img),
                    model=model,
                )
                time_remaining = convert_time_to_float(time_remaining)
        timestamps[str(frame_index)] = {
            "quarter": quarter,
            "time_remaining": time_remaining,
        }
        if frame_index == BREAK:
            break

    return video_path, timestamps


def extract_roi_from_video(video_path: str, model: YOLOModel, device:int=0):
    """
    Find time-remaining roi from video. Assumes static, naive approach.
    Returns a tensor with format: [x1, y1, x2, y2] or None if no
    ROI is found.
    """

    assert os.path.isfile(video_path), f"Error: bad path to video {video_path}."
    # assert video_path[-4:] == '.mp4'

    # print(f"Finding time-remaining ROI for video at {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_remaining_roi = None

    highest_conf = 0.0
    best_roi = None
    step = ROI_STEP

    for i in range(frames_cnt):
        ret, frame = cap.read()
        
        if not ret:
            break
        if i % step == 0:
            results = model.model(frame, verbose=False)
            classes, conf, boxes = (
                results[0].boxes.cls,
                results[0].boxes.conf,
                results[0].boxes.xyxy,
            )
            classes_conf = torch.stack((classes, conf), dim=1)
            predictions = torch.cat((classes_conf, boxes), dim=1)
            conf_mask = predictions[:, 1] > CONF_THRESH
            pred_thresh = predictions[conf_mask]
            for row in pred_thresh:
                if row[0] == QUARTER_KEY:
                    pass
                elif row[0] == TIME_REMAINING_KEY:
                    time_remaining_roi = row[2:].to(torch.int)
            for row in predictions:
                if row[0] == QUARTER_KEY:
                    pass
                elif row[0] == TIME_REMAINING_KEY:
                    if row[1] > highest_conf:
                        highest_conf = row[1]
                        best_roi = row[2:].to(torch.int)
            if time_remaining_roi is not None:
                break
    return best_roi
