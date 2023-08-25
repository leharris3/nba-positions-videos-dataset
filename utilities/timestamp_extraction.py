import cv2
from PIL import Image
import pytesseract
import sys
import os
import numpy as np

# MARK: Change to your local path to tessract.exe
PATH_TO_TESSERACT = r"/usr/local/bin/pytesseract"
OUTPUT_CODEC = "mp4v"
BREAK_POINT = 10
PRINT_FRAME_OFFSET = 50
PRINT_FRAME_OFFSET = 1000
LARGE_STEP = 1000
MOD_STEP = 25
BREAK_POINT = -1
QUARTER_CONFIG = r'--oem 1 --psm 10 -c tessedit_char_whitelist=1234 load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0'
CLOCK_CONFIG = r'--oem 1 --psm 3 -c tessedit_char_whitelist=0123456789.: -c load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0'

ROIS = {
    "TNT_QUARTER": {"x_start": 866, "width": 14, "y_start": 1168-560, "height": 24},
    "TNT_CLOCK": {"x_start": 954, "width": 63, "y_start": 1168-560, "height": 24},
    "ESP_QUARTER": {"x_start": 835, "width": 14, "y_start": 1138-560, "height": 24},
    "ESP_CLOCK": {"x_start": 885, "width": 62, "y_start": 1138-560, "height": 24},
    "FOX_QUARTER": {"x_start": 840, "width": 10, "y_start": 1158-560, "height": 28},
    "FOX_CLOCK": {"x_start": 946, "width": 60, "y_start": 1158-560, "height": 28},
    "CSN_QUARTER": {"x_start": 1005, "width": 129, "y_start": 1183-560, "height": 28},
    "CSN_CLOCK": {"x_start": 1005, "width": 129, "y_start": 1183-560, "height": 28},
    "TSN_QUARTER": {"x_start": 978, "width": 157, "y_start": 1180-560, "height": 23},
    "TSN_CLOCK": {"x_start": 978, "width": 157, "y_start": 1180-560, "height": 23}
}


def extract_timestamps_plus_trim(video_path: str, network: str):
    """Extract timestamps from video. Removes frames w/o valid timestamps and saves as a trimmed video."""

    timestamps = {}
    quarter, clock = f"{network}_QUARTER", f"{network}_CLOCK"
    step, q_width_start, q_width_offset, q_height_start, q_height_offset = LARGE_STEP, *[
        ROIS[quarter][param] for param in ("x_start", "width", "y_start", "height")]
    clk_width_start, clk_width_offset, clk_height_start, clk_height_offset = [
        ROIS[clock][param] for param in ("x_start", "width", "y_start", "height")]

    capture = cv2.VideoCapture(video_path)
    total_frames, output_width, output_height, output_fps = map(int, (
        capture.get(cv2.CAP_PROP_FRAME_COUNT),
        capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        capture.get(cv2.CAP_PROP_FPS)
    ))

    output_path = f"{video_path.strip('.mp4')}_trim.mp4"
    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*OUTPUT_CODEC), output_fps, (output_width, output_height))

    quarter, time_seconds, new_frame_index, first_timestamp_spotted = None, None, 0, False

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
        except:
            raise Exception(
                f"Error: invalid clock or quarter roi provided for video at {video_path}.")

        if (frame_index % step) != 0:
            if quarter is not None and time_seconds is not None:
                timestamps[new_frame_index] = [
                    quarter, time_seconds]
                video_writer.write(frame)
                new_frame_index += 1
        else:
            q_result = preprocess_and_ocr(q_roi, QUARTER_CONFIG)
            clk_result = preprocess_and_ocr(clk_roi, CLOCK_CONFIG)
            if q_result or clk_result:
                print(q_result, clk_result)
            if new_frame_index <= 10:
                cv2.imwrite("q_roi.png", q_result)
                cv2.imwrite("clk_roi.png", clk_result)

            quarter = int(q_result[0]) if q_result else None
            time_seconds = convert_time_to_seconds(clk_result)
            if quarter and time_seconds:
                timestamps[new_frame_index] = [quarter, time_seconds]
                if not first_timestamp_spotted:
                    first_timestamp_spotted = True
                    step = MOD_STEP
                video_writer.write(frame)
                new_frame_index += 1

        if (frame_index % PRINT_FRAME_OFFSET) == 0:
            print_progress(progress=(frame_index + 1) / total_frames)

    video_writer.release()
    capture.release()
    return timestamps, output_path


def preprocess_and_ocr(roi, config):
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

    # if background is black, invert
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
