import os
import sys
from contextlib import contextmanager
import cv2
from ultralytics import YOLO
from tqdm import tqdm

from video import Video
from utilities.text_extraction.entities.roi import ROI
from utilities.text_extraction.timestamp_extraction import is_valid_roi

MODEL_PATH = r"models/yolo/weights/clock_rois_nano.pt"


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def predict_on_frame(frame, model: YOLO):
    """Extract high-confidence roi from frame."""

    results = model(frame, save=True, verbose=False)
    roi = None

    result = results[0]
    confidences = result.boxes.conf
    bounding_boxes = result.boxes.xyxy

    if len(confidences > 0):
        if max(confidences) > .9:
            max_index = confidences.argmax()
            bb = bounding_boxes[max_index]
            bb = [int(val) for val in bb]
            roi = ROI(bb[0], bb[1], bb[2], bb[3], max(confidences))

    return roi


def detect_roi(video: Video):
    """Finds clock roi from a given video."""

    print(f"Loading model at {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Loading complete!")

    video_path = video.get_path()
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        with suppress_stdout():
            roi = predict_on_frame(frame, model)
        if roi and is_valid_roi(frame, roi):
            return roi

    return None
