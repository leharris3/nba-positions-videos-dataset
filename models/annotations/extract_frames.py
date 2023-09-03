import cv2
import os
from tqdm import tqdm

PATH_TO_DATASET = r"C:\Users\Levi\Desktop\basketball-clock-detection-dataset"


def extract_frames(from_path: str, to_path: str):
    assert os.path.exists(from_path)
    assert from_path[-4:] == '.mp4'

    cap = cv2.VideoCapture(from_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    title = os.path.basename(from_path)

    for frame_index in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        digits = len(str(frame_index))
        pad = (6 - digits) * "0"
        cv2.imwrite(f"{to_path}/{title}_frame_{pad}{frame_index}.png", frame)


def extract_labels(from_path: str, to_path: str, video_title: str):
    assert os.path.exists(from_path)

    pass


def add_video_and_labels_to_dataset(labels_path: str, video_path: str, of_type: str):
    """Types: test, train, valid."""

    subdir = ""
    if of_type == "train":
        subdir = "train"
    elif of_type == "test":
        subdir = "test"
    elif of_type == "valid":
        subdir = "valid"
    else:
        raise Exception("No valid type given.")

    labels_dst = f"{PATH_TO_DATASET}/{subdir}/labels"
    frames_dst = f"{PATH_TO_DATASET}/{subdir}/images"
