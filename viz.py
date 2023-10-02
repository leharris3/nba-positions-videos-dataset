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
    font = ImageFont.truetype(r'utilities\os-eb.ttf', 30)  # TODO: FIX

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

def visualize_roi()
