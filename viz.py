import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def visualize_timestamps(video_path, timestamps_path, viz_path):

    with open(timestamps_path, 'r') as f:
        timestamps = json.load(f)

    reader = cv2.VideoCapture(video_path)
    frame_cnt = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width, fps = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(
        cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(
            cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        viz_path, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))
    font = ImageFont.truetype(r'Outilities\os-eb.ttf', 30)

    for frame_index in tqdm(range(frame_cnt)):
        ret, frame = reader.read()
        if not ret:
            break

        quarter, time_remaining = None, None
        index = str(frame_index)
        if index in timestamps:
            quarter = timestamps[str(index)]["quarter"]
            time_remaining = timestamps[str(index)]["time_remaining"]

        img = Image.fromarray(frame)
        draw: ImageDraw = ImageDraw.Draw(img)
        draw.text(
            (10, 10), f"Quarter: {quarter}, Time Remaining: {time_remaining}", font=font, fill=(255, 255, 255))
        writer.write(np.array(img))

    writer.release()
