import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import torch
import json
import os

import pytesseract
import easyocr
from typing import List
from viz import visualize_timestamps

MODEL_PATH = r"models\yolo\weights\45_ep_lar_full.pt"
MODEL = YOLO(MODEL_PATH)

QUARTER_KEY = 0
TIME_REMAINING_KEY = 1

PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
QUARTER_CONFIG = '--oem 3 --psm 7 -c tessedit_char_whitelist=1234 sStTnNdDrRhH'
TIME_REMAINING_CONFIG = '--oem 3 --psm 7'
READER = easyocr.Reader(['en'])
PAD = 3
BREAK = -1


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

    quarter_roi, time_remaining_roi = extract_rois_from_video(video_path)
    q_x1, q_y1, q_x2, q_y2 = quarter_roi
    tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi

    print(f"Extracting timestamps for video at {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = {}

    for frame_index in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        quarter_img = frame[q_y1 - PAD: q_y2 +
                            2 * PAD, q_x1 - PAD: q_x2 + 2 * PAD]
        time_remaining_img = frame[tr_y1 -
                                   PAD: tr_y2 + 2 * PAD, tr_x1 - PAD: tr_x2 + 2 * PAD]

        quarter, time_remaining = extract_timestamps_from_images(
            quarter_img, time_remaining_img, preprocessing_func=preprocess_image_for_tesseract)
        timestamps[str(frame_index)] = {
            "quarter": quarter,
            "time_remaining": time_remaining
        }

        if frame_index == BREAK:
            break

    post_process_timestamps(timestamps)
    with open(save_path, "w") as json_file:
        json.dump(timestamps, json_file, indent=4)


def extract_rois_from_video(path):
    """Find rois from video. Assumes static, naive approach."""

    print(f"Find ROIs for video at {path}")
    cap = cv2.VideoCapture(path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    quarter_roi, time_remaining_roi = None, None

    for _ in tqdm(range(frames_cnt)):
        ret, frame = cap.read()
        if not ret:
            break
        results = MODEL(frame, verbose=False)
        annotated_frame = results[0].plot()

        classes, conf, boxes = results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xyxy
        classes_conf = torch.stack((classes, conf), dim=1)
        predictions = torch.cat((classes_conf, boxes), dim=1)
        conf_mask = predictions[:, 1] > .90
        pred_thresh = predictions[conf_mask]

        for row in pred_thresh:
            if row[0] == QUARTER_KEY:
                quarter_roi = row[2:].to(torch.int)
            elif row[0] == TIME_REMAINING_KEY:
                time_remaining_roi = row[2:].to(torch.int)
        if quarter_roi is not None and time_remaining_roi is not None:
            break

    return quarter_roi, time_remaining_roi


def extract_timestamps_from_images(quarter_image, time_remaining_image, preprocessing_func=None, extraction_method="tesseract"):

    quarter = None
    time_remaining = None

    if extraction_method == "tesseract":
        quarter_r = (extract_text_from_image_with_tesseract(
            quarter_image, preprocess_func=preprocessing_func, config=QUARTER_CONFIG))
        try:
            for res in quarter_r:
                if res != " ":
                    quarter = int(res[0])
        except:
            pass
        time_remaining_r = (extract_text_from_image_with_tesseract(
            time_remaining_image, preprocess_func=preprocessing_func, config=TIME_REMAINING_CONFIG))
        try:
            for res in time_remaining_r:
                if res != " ":
                    time_remaining = convert_time_to_float(res)
                    break
        except:
            pass
    elif extraction_method == "easyocr":
        quarter_r = (extract_text_from_image_with_easyocr(
            quarter_image, preprocess_func=preprocessing_func))
        try:
            for res in quarter_r:
                if res != " ":
                    quarter = int(res[0])
        except:
            pass
        time_remaining_r = (extract_text_from_image_with_easyocr(
            time_remaining_image, preprocess_func=preprocessing_func))
        try:
            for res in time_remaining_r:
                if res != " ":
                    time_remaining = convert_time_to_float(res)
                    break
        except:
            pass

    return quarter, time_remaining


def extract_text_from_image_with_easyocr(image, print_result=None, config=None, preprocess_func=None) -> List[str]:

    extracted_text = []
    results = READER.readtext(
        image, batch_size=16)
    for (_, bb, _) in results:
        extracted_text.append(bb)
        if print_result:
            print(bb)
    return extracted_text


def extract_text_from_image_with_tesseract(image, config="", print_results=None, preprocess_func=None) -> List[str]:

    extracted_text = []
    pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT

    if preprocess_func:
        image = preprocess_func(image)
    results = pytesseract.image_to_string(image, config=config).split("\n")
    for line in results:
        for word in line.split(" "):
            extracted_text.append(word)
    return extracted_text


def convert_time_to_float(time: str) -> float:
    """Convert a formated time str to a float value representing seconds remaining in a basektball game."""

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


def preprocess_image_for_tesseract(image, save=None):
    """Preprocess a ROI for OCR."""

    def change_dpi(image, target_dpi=95):
        """95 is the magic number, font height should be 30-33 px for best results."""

        try:
            image = Image.fromarray(image)
            current_dpi = image.info.get("dpi", (72, 72))
            scale_factor = target_dpi / current_dpi[0]
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            resized_image = image.resize((new_width, new_height))
            resized_image.info["dpi"] = (target_dpi, target_dpi)
            return np.array(resized_image)
        except Exception as e:
            raise Exception("An error while preprocessing a frame:", str(e))

    scaled_image = change_dpi(image)
    return scaled_image


def nparr_from_img_path(img_path):
    """Return a numpy array from an img_path."""

    image = Image.open(img_path)
    return np.array(image)


def post_process_timestamps(timestamps):
    """Interpolate timestamps in-place."""

    last_quarter, last_time = None, None

    for key in timestamps:
        quarter, time_remaining = timestamps[key]["quarter"], timestamps[key]["time_remaining"]
        if not last_quarter:
            if quarter:
                last_quarter = quarter
        else:
            if not quarter:
                timestamps[key]["quarter"] = last_quarter
        if not last_time:
            if time_remaining:
                last_time = time_remaining
        else:
            if not time_remaining:
                timestamps[key]["time_remaining"] = last_time
