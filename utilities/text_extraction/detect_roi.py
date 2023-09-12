import cv2
from ultralytics import YOLO
from tqdm import tqdm

from video import Video
from utilities.text_extraction.entities.roi import ROI

MODEL_PATH = r"models/yolo/weights/clock_rois_nano.pt"


# TODO: currently just saves all prediction vizs.

def detect_roi(video: Video):
    """Finds clock roi from a given video."""

    # Iterate through video frames
    # Predict clock labels
    # If confidence high (> 90): extract text
    # If text found for quarter and time_remaining: return ROI
    # else: keep looking

    print(f"Loading model at {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Loading complete!")

    video_path = video.get_path()
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, save=True)

    return None
