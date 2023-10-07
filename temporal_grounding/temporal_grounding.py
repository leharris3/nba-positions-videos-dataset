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

from temporal_grounding.viz import visualize_timestamps
from temporal_grounding.utilities.models import Models
from temporal_grounding.utilities.constants import *


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

    valid_formats = ['.avi', '.mp4']
    vids = os.listdir(dir_path)
    for vid in vids:
        format_found = False
        for format in valid_formats:
            if format in vid:
                format_found = True
        if not format_found:
            vids.remove(vid)

    for vid in vids:
        video_path = os.path.join(dir_path, vid)
        data_path = os.path.join(data_out_path, vid.replace(
            ".mp4", ".json").replace(".avi", ".json"))
        start_time = time.time()  # for benchmarking purposes
        extract_timestamps_from_video(video_path, data_path)
        if viz_out_path is not None:
            viz_path = os.path.join(
                viz_out_path, vid.replace(".mp4", "_viz.avi"))
            visualize_timestamps(video_path, data_path, viz_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{vid} processed in {elapsed_time:.2f} seconds")


def extract_timestamps_from_video(video_path: str, save_path: str) -> None:
    """
    Given a path to a basketball broadcast video,
    saves a json with extracted timestamp info to save path.
    """

    assert os.path.exists(video_path)
    tr_x1, tr_y1, tr_x2, tr_y2 = None, None, None, None
    time_remaining_roi = extract_roi_from_video(video_path)
    if time_remaining_roi is not None:
        tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi

    print(f"Extracting timestamps for video at {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = {}
    quarter = video_path[-5]  # period_x.mp4
    step = 5

    # TODO: temporarly do not cast quarter to int
    for frame_index in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        time_remaining_img = None
        time_remaining = None
        if frame_index % step == 0:
            if time_remaining_roi is not None:
                time_remaining_img = frame[tr_y1 -
                                           PAD: tr_y2 + 2 * PAD, tr_x1 - PAD: tr_x2 + 2 * PAD]
            if time_remaining_img is not None:
                time_remaining = extract_time_remaining_from_image(
                    Image.fromarray(time_remaining_img))
                time_remaining = convert_time_to_float(time_remaining)
        timestamps[str(frame_index)] = {
            "quarter": quarter,
            "time_remaining": time_remaining,
        }
        if frame_index == BREAK:
            break

    post_process_timestamps(timestamps, fps)

    # TODO: fix! This will break shit!
    timestamps = generate_formated_timestamps(timestamps=timestamps)
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

    print(f"Finding time-remaining ROI for video at {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_remaining_roi = None

    # TODO: skip through vid at one second intervals
    highest_conf = 0.0
    best_roi = None
    step = 30

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

    if image is None:
        return None
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

    # cv2.imwrite("preprocessed_img.png", img_arr)
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


def convert_float_to_time_str(time_remaining: float):
    if time_remaining is not None:
        minutes = int(time_remaining) // 60
        seconds = int((time_remaining - (minutes * 60)))
        decimal_seconds = (
            int((time_remaining - (minutes * 60) - seconds) * 10))
        return f"{minutes:02d}:{seconds:02d}.{decimal_seconds}"
    else:
        return ""


def post_process_timestamps(timestamps, frame_rate: float):
    """
    Interpolate timestamps in-place.
    """

    def interpolate_time_array(time_array, frame_rate: float):
        """
        Interpolate between indentical time values
        only when it is clear that time remaining will 
        decrease in the next one second of frames.
        """

        seen = {}
        seconds_per_frame = 1/frame_rate
        i = 0
        arr_len = len(time_array)

        while i < arr_len:
            interpolate = False
            curr_value = time_array[i]
            peek_index = i + int(frame_rate) - 1
            if not curr_value is None:
                if (peek_index) < arr_len:
                    peek_value = time_array[peek_index]
                    if not peek_value is None and peek_value < curr_value:
                        interpolate = True
                else:
                    interpolate = True
                if interpolate is True:
                    if curr_value in seen:
                        seen[curr_value] += 1
                        time_array[i] -= seen[curr_value] * seconds_per_frame
                    else:
                        seen[curr_value] = 1
                        time_array[i] -= seen[curr_value] * seconds_per_frame
                if time_array[i] < 0:
                    time_array[i] = 0.0
            i += 1

    last_quarter, last_time = None, None
    time_array = []

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
        time_array.append(timestamps[key]["time_remaining"])

    interpolate_time_array(time_array, frame_rate)
    for key, time_remaining in zip(timestamps, time_array):
        if timestamps[key]["time_remaining"] is not None:
            timestamps[key]["time_remaining"] = round(time_remaining, 1)


def generate_formated_timestamps(timestamps):
    """
    Format timestamps to use quarter and time-remaining as keys, 
    frame indices as value.

    "{quarter}_{time_remaining}: frame_index"
    """

    formatted_timestamps = {}
    for key in timestamps:
        values = timestamps[key]
        quarter, time_remaining = values["quarter"], values["time_remaining"]
        frame_index = int(key)
        formatted_timestamps[f"{quarter}_{time_remaining}"] = frame_index
    return formatted_timestamps
