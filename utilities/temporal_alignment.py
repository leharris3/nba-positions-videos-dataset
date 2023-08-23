import cv2
from PIL import Image
import pytesseract
import sys
import os
import numpy as np

from data import Data
from video import Video
from game import Game

# MARK: Change to your local path to tessract.exe
PATH_TO_TESSERACT = r"/usr/local/bin/pytesseract"
PRINT_FRAME_OFFSET = 1
BREAK_POINT = 2
QUARTER_CONFIG = r'--oem 3 --psm 10 -c tessedit_char_whitelist=1234'
CLOCK_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.:'
TRIM_VIDEO_BITRATE = 1000000
TIMESTAMPS = {}

ROIS = {
    "TNT_QUARTER": {"x_start": 866, "width": 14, "y_start": 1168-280, "height": 24},
    "TNT_CLOCK": {"x_start": 954, "width": 63, "y_start": 1168-280, "height": 24},
    "ESP_QUARTER": {"x_start": 835, "width": 13, "y_start": 1138-280, "height": 21},
    "ESP_CLOCK": {"x_start": 885, "width": 59, "y_start": 1138-280, "height": 21},
    "FOX_QUARTER": {"x_start": 840, "width": 10, "y_start": 1158-280, "height": 28},
    "FOX_CLOCK": {"x_start": 946, "width": 60, "y_start": 1158-280, "height": 28},
    "CSN_QUARTER": {"x_start": 1005, "width": 129, "y_start": 1183-280, "height": 28},
    "CSN_CLOCK": {"x_start": 1005, "width": 129, "y_start": 1183-280, "height": 28},
    "TSN_QUARTER": {"x_start": 978, "width": 157, "y_start": 1180-280, "height": 23},
    "TSN_CLOCK": {"x_start": 978, "width": 157, "y_start": 1180-280, "height": 23}
}


def extract_timestamps(game: Game):
    return extract_timestamps_plus_trim(game.video.path, game.network)


def preprocess_image(image):
    """Preprocess a ROI for OCR."""
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3))  # Fix the kernel creation
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close
    return result


def extract_timestamps_plus_trim(video_path: str, network: str):
    """Extract timestamps from video. Removes frames w/o valid timestamps and save as a trimmed video."""

    # pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT
    QUARTER, CLOCK = f"{network}_QUARTER", f"{network}_CLOCK"

    step = 25  # process each step frame
    try:
        q_width_start, q_width_offset, q_height_start, q_height_offset = ROIS[
            QUARTER]["x_start"], ROIS[QUARTER]["width"], ROIS[QUARTER]["y_start"], ROIS[QUARTER]["height"]
        clk_width_start, clk_width_offset, clk_height_start, clk_height_offset = ROIS[
            CLOCK]["x_start"], ROIS[CLOCK]["width"], ROIS[CLOCK]["y_start"], ROIS[CLOCK]["height"]
    except:
        print(f"Error: invalid network: {network}!")
        raise Exception

    capture = cv2.VideoCapture(video_path)
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    quarter, time_seconds = None, None
    new_frame_index = 0
    first_timestamp_spotted = False

    output_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    output_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    output_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = f"{video_path.strip('.mp4')}_trim.mp4"

    try:
        video_writer = cv2.VideoWriter(
            output_path, output_fourcc, output_fps, (output_width, output_height))
    except:
        print(
            f"Error: could not create a video writer for video at {video_path}.")
        raise Exception

    for frame_index in range(total_frames):
        if new_frame_index >= BREAK_POINT:
            break

        ret, frame = capture.read()
        if not ret:
            break

        try:
            q_roi = frame[q_height_start: q_height_start + q_height_offset,
                          q_width_start: q_width_start + q_width_offset]
            clk_roi = frame[clk_height_start: clk_height_start + clk_height_offset,
                            clk_width_start: clk_width_start + clk_width_offset]
        except:
            print(
                f"Error: invalid clock or quarter roi provided for video at {video_path}.")
            raise Exception

        if (frame_index % step) != 0:
            if quarter is not None and time_seconds is not None:
                TIMESTAMPS[new_frame_index] = [
                    int(quarter), time_seconds]
                video_writer.write(frame)
                new_frame_index += 1
        else:

            # Preprocess quarter ROI
            preprocessed_q_roi = preprocess_image(q_roi)
            roi_pil_q = Image.fromarray(preprocessed_q_roi)
            q_result = pytesseract.image_to_string(
                roi_pil_q, config=QUARTER_CONFIG)

            # if q_result.strip() != "":
            #     print("Quarter OCR Result:", q_result)

            # Preprocess clock ROI
            preprocessed_clk_roi = preprocess_image(clk_roi)
            roi_pil_clk = Image.fromarray(preprocessed_clk_roi)
            clk_result = pytesseract.image_to_string(
                roi_pil_clk, config=CLOCK_CONFIG)

            # if clk_result.strip() != "":
            #     print("Clock OCR Result:", clk_result)

            # cv2.imwrite("scripts/quarter.png", np.array((roi_pil_q)))
            # cv2.imwrite("scripts/clock.png", np.array((roi_pil_clk)))
            # exit()

            try:
                quarter = int(q_result[0])
                time_seconds = convert_time_to_seconds(clk_result)
                if quarter is not None and time_seconds is not None:
                    TIMESTAMPS[new_frame_index] = [int(quarter), time_seconds]
                    video_writer.write(frame)
                    new_frame_index += 1
                    if not first_timestamp_spotted:
                        first_timestamp_spotted = True
                        step = 1
            except:
                pass

        if (frame_index % PRINT_FRAME_OFFSET) == 0:
            progress = (frame_index + 1) / total_frames
            print_progress(progress=progress)

    # Release the VideoWriter and capture objects
    video_writer.release()
    capture.release()
    os.remove(video_path)
    os.rename(output_path, video_path)

    return TIMESTAMPS


def convert_time_to_seconds(time_str):
    if ':' in time_str:
        time_parts = time_str.split(':')
        minutes = int(time_parts[0])
        seconds = float(time_parts[1])
    elif '.' in time_str:
        seconds = float(time_str)
        minutes = 0
    else:
        raise ValueError("Invalid time format")
    return minutes * 60 + seconds


def print_progress(progress):
    progress_bar = "[" + "#" * \
        int(progress * 20) + " " * (20 - int(progress * 20)) + "]"
    sys.stdout.write("\r{} {:.2f}%".format(progress_bar, progress * 100))
    sys.stdout.flush()


def save_frames_as_video(frames, output_path, fps=25, bitrate=TRIM_VIDEO_BITRATE):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (width, height), isColor=True)
    out.set(cv2.CAP_PROP_BITRATE, bitrate)
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved successfully at: {output_path}")
