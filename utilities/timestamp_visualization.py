import os
from re import L
import sys
import json
import cv2

from video import Video
from data import Data

FPS_OUT = 25
FRAME_WIDTH_OUT = 1280
FRAME_HEIGHT_OUT = 720
STEP_SIZE = 1


def viz_timestamp_mapping(video: Video, data: Data, out_path):
    """Generate viz from data file with mapped timestamps."""

    assert data.path[-5:] == ".json", "Error: invalid file extension."
    try:
        json_data = read_json(data.path)
    except:
        raise Exception(f"Error: could not read in data from {data.path}.")
    try:
        video_capture = cv2.VideoCapture(video.path)
    except:
        raise Exception(f"Error: could not open video at path: {video.path}.")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, FPS_OUT,
                              (FRAME_WIDTH_OUT, FRAME_HEIGHT_OUT))
    except:
        raise Exception(
            f"Error: could not generate a video writer for out path: {out_path}.")

    for frame_index in data.get_frames_moments_mapped():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = video_capture.read()
        if not ret:
            break
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    total_frames = len(json_data.items())
    for i, (frame_number, data) in enumerate(json_data.items()):
        if i % STEP_SIZE == 0:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = video_capture.read()
            if not ret:
                break
            quarter, seconds_remaining = data[0], data[1]
            time_str = format_time(seconds_remaining)
            text_overlay = f"Quarter: {quarter} -- Time Remaining: {time_str}"
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, text_overlay, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)
        progress = (i + 1) / total_frames
        sys.stdout.write('\r')
        sys.stdout.write(
            f"Processing frames: [{'=' * int(30 * progress):<30}] {round((progress * 100), 2)}%")
        sys.stdout.flush()
    video_capture.release()
    out.release()

    assert os.path.exists(out_path)
    print(f"Generated timestamp visualization at: {out_path}.")


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def read_json(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except:
        print(f"Error: could not read in data from {file_path}.")
        raise Exception
