import cv2
import torch
import json
import os
import re
import numpy as np

from paddleocr import PaddleOCR
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from copy import copy
from typing import List

from viz import visualize_timestamps

MODEL_PATH = r"models\yolo\weights\45_ep_lar_full.pt"
MODEL = YOLO(MODEL_PATH)

QUARTER_KEY = 0
TIME_REMAINING_KEY = 1

PAD = 3
BREAK = -1
CONF_THRESH = .88
OCR = PaddleOCR(use_angle_cls=True,
                lang='en',
                show_log=False,
                det_db_score_mode='slow',
                ocr_version='PP-OCRv4',
                rec_algorithm='SVTR_LCNet',
                drop_score=0.9,
                )


def process_dir(dir):

    vids = os.listdir(dir)
    data_path = r"timestamps\data"
    vizs = r"timestamps\vizs"

    for vid in vids:
        video_path = os.path.join(dir, vid)
        save_path = os.path.join(data_path, vid.replace(".mp4", ".json"))
        viz_path = os.path.join(vizs, vid.replace(".mp4", "_viz.avi"))
        extract_timestamps_from_video(video_path, save_path)
        visualize_timestamps(video_path, save_path, viz_path)


def extract_timestamps_from_video(video_path, save_path):

    time_remaining_roi = extract_roi_from_video(video_path)

    if time_remaining_roi is not None:
        tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi

    print(f"Extracting timestamps for video at {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = {}
    quarter = video_path[-5]  # period_x.mp4

    for frame_index in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break

        time_remaining_img = None
        if time_remaining_roi is not None:
            time_remaining_img = frame[tr_y1 -
                                       PAD: tr_y2 + 2 * PAD, tr_x1 - PAD: tr_x2 + 2 * PAD]

        if not time_remaining_img is None:
            time_remaining = extract_time_remaining_from_image(
                Image.fromarray(time_remaining_img))
            time_remaining = convert_time_to_float(time_remaining)

        timestamps[str(frame_index)] = {
            "quarter": quarter,
            "time_remaining": time_remaining
        }
        if frame_index == BREAK:
            break

    post_process_timestamps(timestamps)
    with open(save_path, "w") as json_file:
        json.dump(timestamps, json_file, indent=4)


def extract_roi_from_video(path):
    """Find time-remaining roi from video. Assumes static, naive approach."""

    print(f"Finding time-remaining ROI for video at {path}")

    cap = cv2.VideoCapture(path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_remaining_roi = None

    for i in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        results = MODEL(frame, verbose=False)

        classes, conf, boxes = results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xyxy
        classes_conf = torch.stack((classes, conf), dim=1)
        predictions = torch.cat((classes_conf, boxes), dim=1)
        conf_mask = predictions[:, 1] > CONF_THRESH
        pred_thresh = predictions[conf_mask]

        for row in pred_thresh:
            if row[0] == QUARTER_KEY:
                pass
            elif row[0] == TIME_REMAINING_KEY:
                time_remaining_roi = row[2:].to(torch.int)
        if time_remaining_roi is not None:
            break

    return time_remaining_roi


def extract_time_remaining_from_image(image: Image):

    def find_time_remaining_from_results(results):

        # matches any string showing a valid time remaining of 20 minutes or less
        # assumes brodcasts use MM:SS for times > 1 minute, and SS.S for times < 1 minute

        time_remaining_regex = "(20:00)|(1[0-9]|[0-5]?[0-9]):[0-5][0-9]|[0-5]?[0-9].[0-9]"
        for result in results:
            result = result.replace(" ", "")
            match = re.match(time_remaining_regex, result)
            if match is not None and len(match[0]) == len(result):
                return result
        return None

    results = extract_text_with_paddle(
        image)
    time_remaining = find_time_remaining_from_results(results)
    return time_remaining


def extract_text_with_paddle(image: Image):

    # Preprocess.
    scale_factor = 2
    new_size = (image.width * scale_factor,
                image.height * scale_factor)
    image = image.resize(new_size)

    results = []
    raw_result = OCR.ocr(np.array(image), cls=True)
    for idx in range(len(raw_result)):
        res = raw_result[idx]
        if res is not None:
            for line in res:
                results.append(line[1][0])
    return results


def convert_time_to_float(time: str) -> float:
    """Convert a formated time str to a float value representing seconds remaining in a basektball game."""

    if time is None:
        return None
    result: float = 0.0

    if ':' in time:
        time_arr = time.split(':')
        minutes = time_arr[0]
        seconds = time_arr[1]
        result = (int(minutes) * 60) + int(seconds)
    elif '.' in time:
        time_arr = time.split('.')
        seconds = time_arr[0]
        milliseconds = time_arr[1]
        result = int(seconds) + float('.' + milliseconds)
    else:
        raise Exception(
            f"Error: Invalid format provided for seconds remaining.")
    return result


def preprocess_image_for_paddel(image):
    pass


def nparr_from_img_path(img_path):
    """Return a numpy array from an img_path."""

    image = Image.open(img_path)
    return np.array(image)


def post_process_timestamps(timestamps):
    """Interpolate timestamps in-place."""

    last_quarter, last_time = None, None

    for key in timestamps:
        quarter, time_remaining = timestamps[key]["quarter"], timestamps[key]["time_remaining"]
        if quarter:
            last_quarter = quarter
        else:
            timestamps[key]["quarter"] = last_quarter
        if time_remaining:
            last_time = time_remaining
        else:
            timestamps[key]["time_remaining"] = last_time
