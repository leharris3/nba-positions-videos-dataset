import cv2
import torch
import json
import os
import re
import numpy as np

from tqdm import tqdm
from PIL import Image
from typing import List

from viz import visualize_timestamps
from utilities.models import Models
from utilities.constants import *

MODELS = Models()


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
    if not MODELS.models_loaded:
        MODELS.load()
    cap = cv2.VideoCapture(path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_remaining_roi = None

    for i in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        results = MODELS.yolo(frame, verbose=False)

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


def find_time_remaining_from_results(results):
    """Matches any string showing a valid time remaining of 20 minutes or less
    assumes brodcasts use MM:SS for times > 1 minute, and SS.S for times < 1 minute
    """
    if results is None:
        return None
    time_remaining_regex = "(20:00)|(0[0-9]?:[0-9][0-9](\.[0-9])?)|([1-9]:[0-5][0-9])|(1[0-9]:[0-5][0-9](\.[0-9])?)|([0-9]\.[0-9])|([1-5][0-9]\.[0-9])"
    for result in results:
        result = result.replace(" ", "")
        match = re.match(time_remaining_regex, result)
        if match is not None and match[0] == result:
            return result
    return None


def extract_time_remaining_from_image(image: Image):

    results = extract_text_with_paddle(
        image)
    time_remaining = find_time_remaining_from_results(results)
    return time_remaining


def extract_text_with_paddle(image):

    if image is None:
        return []
    if not MODELS.models_loaded:
        MODELS.load()

    ideal_height = 75
    scale_factor = ideal_height / image.height
    new_size = (int(image.width * scale_factor),
                int(image.height * scale_factor))

    image = np.array(image.resize(new_size))
    cv2.imwrite("preprocessed_img.png", image)

    results = []
    raw_result = MODELS.paddle_ocr(image)
    text_arr = raw_result[1]
    for pred in text_arr:
        word = pred[0]
        results.append(word)
    return results


def convert_time_to_float(time):
    """Convert a formated time str to a float value representing seconds remaining in a basektball game."""

    if time is None:
        return None

    minutes, seconds = 0.0, 0.0
    if ':' in time:
        time_arr = time.split(':')
        minutes = float(time_arr[0])
        seconds = float(time_arr[1])
    elif '.' in time:
        seconds = float(time)
    else:
        return None
    return (60.0 * minutes) + seconds


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
