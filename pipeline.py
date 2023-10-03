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


def process_dir(dir_path: str, data_out_path: str, viz_out_path=None) -> None:
    """
    Extract all timestamps in a directory.
    Save timestamps and optional visualizations.
    """

    assert os.path.isdir(
        dir_path), f"Error: bad path to video directory: {dir_path}"
    os.makedirs(data_out_path, exist_ok=True)
    if viz_out_path is not None:
        assert type(
            viz_out_path) is str, "Error: path links must be of type str."
        os.makedirs(viz_out_path, exist_ok=True)

    vids = os.listdir(dir_path)
    for vid in vids:
        video_path = os.path.join(dir_path, vid)
        save_path = os.path.join(data_out_path, vid.replace(".mp4", ".json"))
        viz_path = os.path.join(viz_out_path, vid.replace(".mp4", "_viz.avi"))
        extract_timestamps_from_video(video_path, save_path)
        if viz_out_path is not None:
            visualize_timestamps(video_path, save_path, viz_path)


def extract_timestamps_from_video(video_path: str, save_path: str) -> None:
    """
    Given a path to a basketball broadcast video,
    saves a json with extracted timestamp info to save path.
    """

    assert os.path.isfile(video_path)
    assert not os.path.isfile(
        save_path), f"Warning, overwriting exisiting file at {save_path}"
    assert video_path[-4:
                      ] == '.mp4', f"Bad video format: {video_path}, must be an mp4."

    print(f"Extracting timestamps for video at {video_path}")
    time_remaining_roi = extract_roi_from_video(video_path)
    if time_remaining_roi is not None:
        tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = {}
    quarter = video_path[-5]  # period_x.mp4

    for frame_index in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        time_remaining_img = None
        time_remaining = None
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


def extract_roi_from_video(video_path: str):
    """
    Find time-remaining roi from video. Assumes static, naive approach.
    Returns a tensor with format: [x1, y1, x2, y2] or None if no
    ROI is found.
    """

    assert os.path.isfile(
        video_path), f"Error: bad path to video {video_path}."
    assert video_path[-4:] == '.mp4'

    print(f"Finding time-remaining ROI for video at {video_path}")
    if not MODELS.models_loaded:
        MODELS.load()
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_remaining_roi = None

    # TODO: skip through vid at one second intervals
    highest_conf = 0.0
    best_roi = None
    interval = 30

    for i in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        if (i % interval == 0):
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


def find_time_remaining_from_results(results: [str]):
    """
    Matches any string showing a valid time remaining of 20 minutes or less
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
    """
    Given a PIL Image object,
    returns either a valid formatted time-remaining str (ie '11:30')
    or None.
    """
    results = extract_text_with_paddle(
        image)
    time_remaining = find_time_remaining_from_results(results)
    return time_remaining


def extract_text_with_paddle(image) -> [str]:
    """
    Returns a [str] containing all words found in a
    provided PIL image.
    """

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


def convert_time_to_float(time_remaining: str):
    """
    Coverts valid time-remaining str
    to float value representation.
    Return None if time-remaining is invalid.

    Ex: '1:30' -> 90.
    """

    if time_remaining is None:
        return None
    minutes, seconds = 0.0, 0.0
    if ':' in time_remaining:
        time_arr = time_remaining.split(':')
        minutes = float(time_arr[0])
        seconds = float(time_arr[1])
    elif '.' in time_remaining:
        seconds = float(time_remaining)
    else:
        return None
    return (60.0 * minutes) + seconds


def post_process_timestamps(timestamps):
    """
    Interpolate timestamps in-place.
    """

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
