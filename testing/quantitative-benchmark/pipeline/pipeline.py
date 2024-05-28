import time
import concurrent.futures
import cv2
import torch
import os
import concurrent
import logging
import subprocess
import shutil

from tqdm import tqdm
from transformers import logging

from utils.constants import QUARTER_KEY, TIME_REMAINING_KEY, BREAK, CONF_THRESH
from ocr.helpers import convert_time_to_float, find_time_remaining_from_results
from ocr.models import YOLOModel

logging.set_verbosity_error()

MAX_GPUS = 8
ROI_STEP = 5
TIME_REMAINING_STEP = 5

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
    vids.sort()

    mp4_vids = []
    for vid in vids:
        for format in valid_formats:
            if vid.endswith(format):
                mp4_vids.append(vid)
                break

    vids = mp4_vids

    # NBA video idxs.
    vids = vids[0: 2648]

    timestamps = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_GPUS) as executor:
        with tqdm(total=len(vids), desc="Processing Videos") as pbar:
            while len(vids) > 0:
                processes = []
                video_paths = []
                for device in range(MAX_GPUS):
                    if len(vids) == 0:
                        break
                    video_path = os.path.join(dir_path, vids[0])

                    process = executor.submit(
                        extract_timestamps_from_video,
                        video_path,
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
    tr_x1, tr_y1, tr_x2, tr_y2 = None, None, None, None

    # create ROT det. model
    if str(device) not in ROI_MODELS:
        model = YOLOModel(device=device)
        ROI_MODELS[str(device)] = model
    yolo_model = ROI_MODELS[str(device)]

    time_remaining_roi = extract_roi_from_video(video_path, yolo_model, device=device)
    if time_remaining_roi is not None:
        tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi
    timestamps = {}
    quarter = video_path[-5]  # period_x.mp4

    temp_name = f"temp_{os.path.basename(video_path)}"
    os.mkdir(temp_name)
    temp_dir_path = os.path.join(os.getcwd(), temp_name)

    def save_frame(image, path):
        original_height, original_width = image.shape[:2]
        new_height = 100
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height))
        cv2.imwrite(path, resized_image)

    def save_all_images(vid_path: str, dst_dir: str):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Error opening video file: {vid_path}")
            return
        
        frame_number = 0
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            while frame_number < frame_cnt:
                ret, frame = cap.read()
                if frame_number % TIME_REMAINING_STEP != 0:
                    frame_number += 1
                    continue
                if not ret:
                    break
                frame = frame[tr_y1:tr_y2, tr_x1:tr_x2]
                frame_filename = os.path.join(dst_dir, f"{frame_number:05d}.png")
                executor.submit(save_frame, frame, frame_filename)
                frame_number += 1
        
        cap.release()
        # print(f"Saved {frame_number} frames to {dst_dir}")

    # save all frames to a temp dir
    save_all_images(video_path, temp_dir_path)

    paddle_dir = "/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/pipeline/PaddleOCR"
    os.chdir(paddle_dir)

    # batch infer w/ paddel
    predict_command = [
        "python3", "tools/infer/predict_rec.py",
        f"--image_dir={temp_dir_path}",
        "--rec_model_dir=./en_PP-OCRv4_rec_infer/",
        "--rec_char_dict_path=ppocr/utils/en_dict.txt",
        "--use_gpu=True",
    ]

    result = subprocess.run(predict_command, capture_output=True, text=True)
    shutil.rmtree(temp_dir_path)

    output = result.stdout
    output_arr = output.split(":('")
    preds = [x.split('\n[')[0].replace(")", "").replace("'", "").split(", ") for x in output_arr][1:]

    def interpolate_missing_frames(preds):

        preds = preds.copy()
        preds_extended = []
        last = None

        i = 0
        for curr in preds:
            curr = preds[i]
            if i % TIME_REMAINING_STEP == 0:
                last = [curr]
                preds_extended.append(curr)
                i += 1
                continue
            else:
                # print(last * (ROI_STEP - 1))
                for item in last * (ROI_STEP):
                    preds_extended.append(item)

        return preds_extended

    preds = interpolate_missing_frames(preds)

    for frame_idx, pred in enumerate(preds):
        time_remaining, conf = pred[0], pred[1]
        time_remaining = find_time_remaining_from_results([time_remaining])
        time_remaining = convert_time_to_float(time_remaining)
        timestamps[str(frame_idx)] = {
            "quarter": quarter,
            "time_remaining": time_remaining,
            "conf": conf
        }
        if frame_idx == BREAK:
            break

    return video_path, timestamps


def extract_roi_from_video(video_path: str, model: YOLOModel, device:int=0):
    """
    Find time-remaining roi from video. Assumes static, naive approach.
    Returns a tensor with format: [x1, y1, x2, y2] or None if no
    ROI is found.
    """

    assert os.path.isfile(video_path), f"Error: bad path to video {video_path}."
    
    cap = cv2.VideoCapture(video_path)
    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_remaining_roi = None

    highest_conf = 0.0
    best_roi = None
    step = ROI_STEP

    # TODO: batch process ROIs
    start = time.time()
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
    end = time.time()
    # print(f"ROI extraction time: {end - start}")
    return best_roi