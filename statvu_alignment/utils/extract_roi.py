import cv2
import torch
import os

from utils._constants import QUARTER_KEY, TIME_REMAINING_KEY, CONF_THRESH
from utils._models import YOLOModel

MAX_THREADS = 8
MAX_GPUS = 8

ROI_STEP = 30
ROI_MAX_BATCH_SIZE = 1000

TIME_REMAINING_STEP = 3

ROI_MODELS = {}
MODELS = {}

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