import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from copy import copy
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
TIME_REMAINING_CONFIG = '--oem 3 --psm 7 -c tessedit_char_whitelist=1234567890:. '
READER = easyocr.Reader(['en'])
PAD = 0
BREAK = -1
CONF_THRESH = .92


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
        time_remaining = extract_timestamps_from_images(
            time_remaining_img, preprocessing_func=preprocess_image_for_tesseract)
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

    for _ in tqdm(range(frames_cnt)):
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


def extract_timestamps_from_images(time_remaining_image, preprocessing_func=None, extraction_method="tesseract"):

    time_remaining = None

    if extraction_method == "tesseract":
        time_remaining_r = (extract_text_from_image_with_tesseract(
            time_remaining_image, preprocess_func=preprocessing_func, config=TIME_REMAINING_CONFIG))
        try:
            for res in time_remaining_r:
                if res != " ":
                    time_remaining = convert_time_to_float(res)
                    break
        except:
            pass

    return time_remaining


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

    if image is None:
        return None

    extracted_text = []
    pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT

    if preprocess_func:
        image = preprocess_func(image)
    cv2.imwrite("timestamps/clk.png", image)

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
    # gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)[1]
    # kernel = np.ones((1, 1), np.uint8)
    # result = cv2.dilate(thresh, kernel, iterations=2)
    # result = thresh

    # result_c1 = copy(result)
    # result_c2 = copy(result)

    # black_pixels = result_c1[np.where(result_c1 == 0)].size
    # white_pixels = result_c2[np.where(result_c2 == 255)].size

    # if black_pixels > white_pixels:
    #     result = cv2.bitwise_not(result)

    # if type(save) is bool and save:
    #     out_path = f"{SAVE_PATH}/pp.png"
    #     cv2.imwrite(out_path, result)

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
        if quarter:
            last_quarter = quarter
        else:
            timestamps[key]["quarter"] = last_quarter
        if time_remaining:
            last_time = time_remaining
        else:
            timestamps[key]["time_remaining"] = last_time
