import cv2
import torch
import json
import os
import re
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
from viz import visualize_timestamps
from utilities.models import Models
from utilities.constants import *

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 16

def process_video(video, dir_path, data_out_path, viz_out_path):
    """
    Process a single video file.
    """
    try:
        video_path = os.path.join(dir_path, video)
        data_path = os.path.join(data_out_path, video.replace(".mp4", ".json").replace(".avi", ".json"))
        start_time = time.time()  # for benchmarking purposes
        extract_timestamps_from_video(video_path, data_path)
        if viz_out_path is not None:
            viz_path = os.path.join(viz_out_path, video.replace(".mp4", "_viz.avi"))
            visualize_timestamps(video_path, data_path, viz_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{video} processed in {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing {video}: {e}")


def process_dir(dir_path: str, data_out_path: str, viz_out_path=None, preprocessed_videos=None) -> None:
    """
    Process a directory of videos concurrently.
    """
    assert os.path.isdir(dir_path), f"Error: bad path to video directory: {dir_path}"
    os.makedirs(data_out_path, exist_ok=True)
    if viz_out_path is not None:
        assert isinstance(viz_out_path, str), "Error: path links must be of type str."
        os.makedirs(viz_out_path, exist_ok=True)

    valid_formats = {'avi', 'mp4'}
    vids = [vid for vid in os.listdir(dir_path) if vid.split(".")[-1] in valid_formats]

    preprocessed_set = set()
    if preprocessed_videos is not None:
        with open(preprocessed_videos, 'r') as f:
            preprocessed_set = set(f.read().splitlines())
            preprocessed_set = {line.replace('_viz.avi', '.mp4') for line in preprocessed_set}

    vids = [vid for vid in vids if vid not in preprocessed_set]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_video, vid, dir_path, data_out_path, viz_out_path) for vid in vids]
        for future in as_completed(futures):
            pass


def extract_timestamps_from_video(video_path: str, save_path: str) -> None:
    """
    Given a path to a basketball broadcast video,
    saves a json with extracted timestamp info to save path.
    """

    assert os.path.exists(video_path)
    time_remaining_roi = extract_roi_from_video(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video at: {video_path}")
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    quarter = video_path[-5]  # period_x.mp4
    step = 5

    print(f"Extracting timestamps for video at {video_path}")
    timestamps = {}
    for frame_index in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            time_remaining = None
            if time_remaining_roi is not None:
                tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi
                time_remaining_img = frame[
                    tr_y1 - PAD: tr_y2 + 2 * PAD,
                    tr_x1 - PAD: tr_x2 + 2 * PAD
                ]
                if time_remaining_img is not None:
                    time_remaining = extract_time_remaining_from_image(
                        Image.fromarray(time_remaining_img)
                    )
                    time_remaining = convert_time_to_float(time_remaining)
        timestamps[str(frame_index)] = {
            'quarter': quarter,
            'time_remaining': time_remaining
        }
    cap.release()

    post_process_timestamps(timestamps)
    with open(save_path, "w") as json_file:
        json.dump(timestamps, json_file, indent=4)


def extract_roi_from_video(video_path: str):
    """
    Find time-remaining roi from video. Assumes static, naive approach.
    Returns a tensor with format: [x1, y1, x2, y2] or None if no
    ROI is found.
    """

    SKIP_SECONDS = 30
    FPS = 30

    assert os.path.isfile(
        video_path), f"Error: bad path to video {video_path}."
    # assert video_path[-4:] == '.mp4'

    print(f"Finding time-remaining ROI for video at {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_remaining_roi = None

    # TODO: skip through vid at one second intervals
    highest_conf = 0.0
    best_roi = None
    step = SKIP_SECONDS * FPS

    for i in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        if (i % step == 0):
            results = Models.yolo(frame, verbose=False)
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


def find_time_remaining_from_results(results: List[str]):
    """
    Matches any string showing a valid time remaining of 20 minutes or less
    assumes brodcasts use MM:SS for times > 1 minute, and SS.S for times < 1 minute
    """
    if results is None:
        return None
    time_remaining_regex = r"(20:00)|(0[0-9]?:[0-9][0-9](\.[0-9])?)|([1-9]:[0-5][0-9])|(1[0-9]:[0-5][0-9](\.[0-9])?)|([0-9]\.[0-9])|([1-5][0-9]\.[0-9])"
    for result in results:
        result = result.replace(" ", "")
        match = re.match(time_remaining_regex, result)
        if match is not None and match[0] == result:
            return result
    return None


def extract_time_remaining_from_image(image: Image.Image):
    """
    Given a PIL Image object,
    returns either a valid formatted time-remaining str (ie '11:30')
    or None.
    """
    rgb_img = None
    if image is not None:
        rgb_img = image.convert("RGB")
    results = extract_text_with_paddle(
        rgb_img)
    time_remaining = find_time_remaining_from_results(results)
    return time_remaining


def extract_text_with_paddle(image: Image.Image) -> List[str]:
    """
    Returns a [str] containing all words found in a
    provided PIL image.
    """

    if image is None:
        return []
    image = image.convert("RGB")
    ideal_height = 100
    scale_factor = ideal_height / image.height
    new_size = (int(image.width * scale_factor),
                int(image.height * scale_factor))
    image = image.resize(new_size)
    img_arr = np.array(image)
    results = []
    raw_result = Models.paddle_ocr(img_arr)
    text_arr = raw_result[1]
    for pred in text_arr:
        word = pred[0]
        results.append(word)
    return results


def convert_time_to_float(time_remaining):
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

    def interpolate(data):

        time_remaining = []
        for k, v in data.items():
            tr = v['time_remaining']
            time_remaining.append(tr) if tr != None else time_remaining.append(0)

        fps = 30
        multiplier = 0
        decreasing = False
        last_index = len(time_remaining)
        for i in range(last_index):
            curr = time_remaining[i]
            peak_value = time_remaining[i]
            if curr == 0:
                continue
            if decreasing:
                if multiplier == 30:
                    multiplier = 0
                    decreasing = False
                else:
                    time_remaining[i] -= round(((1/30) * multiplier), 2)
                    multiplier += 1
                    continue
            if i < (last_index - fps):
                peak_value = time_remaining[i + fps]
            if peak_value < curr:
                decreasing = True
            else:
                decreasing = False
            if not decreasing:
                time_remaining[i] = 0
        
        for k, v in data.items():
            data[k]['time_remaining'] = time_remaining[int(k)] if time_remaining[int(k)] != 0 else None
        return data

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

    return timestamps
