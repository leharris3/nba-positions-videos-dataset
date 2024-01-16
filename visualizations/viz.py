import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

CHUNK_SIZE = 100
FONT = ImageFont.truetype('utilities/os-eb.ttf', 30)

def visualize_timestamps(input_path, timestamps_path, output_path, tr_roi=None):
    print(f"Generating visualization for video at: {input_path}")
    with open(timestamps_path, 'r') as f:
        timestamps = json.load(f)

    reader = cv2.VideoCapture(input_path)
    if not reader.isOpened():
        print("Error: Could not open input video.")
        return

    frame_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change the codec if needed
    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    new_width, new_height = 480, 360
    writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    font = ImageFont.truetype('utilities/os-eb.ttf', 30)

    for frame_index in tqdm(range(total_frames)):
        ret, frame = reader.read()
        if not ret:
            break

        # resize frame
        frame = cv2.resize(frame, (new_width, new_height))
        quarter, time_remaining = None, None
        minutes, seconds, decimal_seconds = -1, -1, -1
        index = str(frame_index)
        if index in timestamps:
            quarter = timestamps[str(index)]["quarter"]
            time_remaining = timestamps[str(index)]["time_remaining"]
            if time_remaining is not None:
                minutes = int(time_remaining) // 60
                seconds = int(time_remaining - (minutes * 60))
                decimal_seconds = int(
                    (time_remaining - minutes * 60 - seconds) * 10)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text(
            (10, 10), text=f"Q: {quarter} T: {minutes:}:{seconds:}.{decimal_seconds}", font=font, fill=(255, 255, 255))
        frame = np.array(img)
        writer.write(frame)

    reader.release()
    writer.release()


def visualize_roi(video_path, viz_path, roi):

    reader = cv2.VideoCapture(video_path)
    frame_cnt = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width, fps = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(
        cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        viz_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    x1, y1, x2, y2 = None, None, None, None
    if roi is not None:
        x1, y1, x2, y2 = roi.tolist()

    color = (0, 0, 255)
    thickness = 2
    for _ in tqdm(range(frame_cnt)):
        ret, frame = reader.read()
        if not ret:
            break

        if x1 is not None:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        writer.write(frame)

    writer.release()
    reader.release()
