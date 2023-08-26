import cv2
import pytesseract
import sys
import os
import numpy as np
from PIL import Image

from utilities.timestamp_constants import *

# TODO: Rigorously test


def extract_timestamps(video_path: str, network: str) -> dict:
    """Extract timestamps from video."""

    timestamps = {}
    quarter_key = f"{network}_QUARTER"
    clock_key = f"{network}_CLOCK"

    q_params = ROIS[quarter_key]
    clk_params = ROIS[clock_key]

    q_width_start, q_width_offset, q_height_start, q_height_offset = q_params[
        "x_start"], q_params["width"], q_params["y_start"], q_params["height"]
    clk_width_start, clk_width_offset, clk_height_start, clk_height_offset = clk_params[
        "x_start"], clk_params["width"], clk_params["y_start"], clk_params["height"]

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    quarter, time_seconds = None, None
    new_frame_index = 0
    first_timestamp_spotted = False
    step = LARGE_STEP

    for frame_index in range(total_frames):
        if new_frame_index == BREAK_POINT:
            break
        ret, frame = capture.read()
        if not ret:
            break
        try:
            q_roi = frame[q_height_start: q_height_start + q_height_offset,
                          q_width_start: q_width_start + q_width_offset]
            clk_roi = frame[clk_height_start: clk_height_start + clk_height_offset,
                            clk_width_start: clk_width_start + clk_width_offset]
        except IndexError:
            raise Exception(
                f"Error: invalid clock or quarter ROI provided for video at {video_path}.")

        if frame_index % step == 0:
            # Process ROIs with Tesseract
            q_result = preprocess_and_ocr(q_roi, QUARTER_CONFIG)
            clk_result = preprocess_and_ocr(clk_roi, CLOCK_CONFIG)
            time_seconds = convert_time_to_seconds(clk_result)
            quarter = int(q_result[0]) if q_result else None
            if quarter and time_seconds:
                timestamps[new_frame_index] = [quarter, time_seconds]
                if not first_timestamp_spotted:
                    first_timestamp_spotted = True
                    step = MOD_STEP
                new_frame_index += 1
        else:
            # Use last valid timestamp value
            timestamps[new_frame_index] = [
                quarter, time_seconds]
            new_frame_index += 1

        if frame_index % PRINT_FRAME_OFFSET == 0:
            print_progress(progress=(frame_index + 1) / total_frames)

    capture.release()
    return timestamps


def preprocess_and_ocr(roi, config):
    """Return OCR result from an image and a tesseract config."""

    preprocessed_roi = preprocess_image(roi)
    roi_pil = Image.fromarray(preprocessed_roi)
    result = pytesseract.image_to_string(roi_pil, config=config).strip()
    return result


def preprocess_image(image):
    """Preprocess a ROI for OCR."""

    # 95 is the magic number, font height should be 30-33 px for best results.
    def change_dpi(image, target_dpi=95):
        try:
            image = Image.fromarray(image)
            current_dpi = image.info.get("dpi", (72, 72))
            scale_factor = target_dpi / current_dpi[0]
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            resized_image = image.resize((new_width, new_height))
            resized_image.info["dpi"] = (target_dpi, target_dpi)
            return np.array(resized_image)
        except Exception as e:
            print("An error while preprocessing a frame:", str(e))
            raise Exception

    scaled_image = change_dpi(image)
    thresh = cv2.threshold(scaled_image, 135, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3))  # Fix the kernel creation
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close
    top_right_pixel = result[0, -1]
    if top_right_pixel[0] == 0:
        result = cv2.bitwise_not(result)
    return result


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
