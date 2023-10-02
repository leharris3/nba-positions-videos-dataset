import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# TODO: also viz bounding box representing roi

def visualize_timestamps(video_path, timestamps_path, viz_path, tr_roi=None):

    print(f"Generating visualization for video at: {video_path}")

    with open(timestamps_path, 'r') as f:
        timestamps = json.load(f)

    reader = cv2.VideoCapture(video_path)
    frame_cnt = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width, fps = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(
        cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(
            cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        viz_path, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))
    font = ImageFont.truetype(r'utilities\os-eb.ttf', 30) \

    for frame_index in tqdm(range(frame_cnt)):
        ret, frame = reader.read()
        if not ret:
            break

        quarter, time_remaining = None, None
        minutes, seconds = None, None
        index = str(frame_index)
        if index in timestamps:
            quarter = timestamps[str(index)]["quarter"]
            time_remaining = timestamps[str(index)]["time_remaining"]
            if time_remaining is not None:
                minutes = time_remaining // 60
                seconds = round(time_remaining % 60, 2)

        img = Image.fromarray(frame)
        draw: ImageDraw = ImageDraw.Draw(img)
        draw.text(
            (10, 10), f"Quarter: {quarter} \nTime Remaining: {minutes}:{seconds}", font=font, fill=(255, 255, 255))
        writer.write(np.array(img))

    writer.release()


def visualize_roi(video_path, viz_path, roi):

    reader = cv2.VideoCapture(video_path)
    frame_cnt = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width, fps = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(
        cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(
            cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        viz_path, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

    x1, y1, x2, y2 = None, None, None, None
    if roi is not None:
        x1, y1, x2, y2 = roi.tolist()
        print(x1, y1, x2, y2)

    color = (0, 0, 255)
    thickness = 2
    for _ in tqdm(range(frame_cnt)):
        ret, frame = reader.read()
        if not ret:
            break

        if x1 is not None:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        writer.write(np.array(frame))

    writer.release()
    reader.release()
