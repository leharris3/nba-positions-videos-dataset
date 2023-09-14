import sys
import cv2
import os
import pytesseract
from video import Video
from utilities.text_extraction.entities.roi import ROI
from utilities.files import File
from utilities.text_extraction.preprocessing import preprocess_image

PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class FrameTimestamp:

    def __init__(self, quarter=None, time_remaining=None) -> None:
        self.quarter: int = quarter
        self.time_remaining: float = time_remaining


class VideoTimestamps:

    def __init__(self, video=None) -> None:
        self.timestamps = {}
        self.video: Video = video

    def set_timestamp(self, frame_index: int, frame_timestamp: FrameTimestamp):

        timestamp = {'quarter': frame_timestamp.quarter,
                     'time_remaining': frame_timestamp.time_remaining}
        self.timestamps[frame_index] = timestamp

    def save_timestamps_to(self, path):

        assert not os.path.exists, f"Error: file found at path: {path}"
        File.save_json(self.timestamps, to=path)


def extract_timestamps_from_video(video: Video):
    """Return a dict {frame: [quarter, time_remaining]}"""

    return {}


def extract_timestamps_from_frame(frame) -> FrameTimestamp:
    """Return a dict {quarter: int | None, time_remaining: float | None} from a frame."""

    # Optional: append path to tesseract to sys.
    pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT

    # TODO: TEMP
    frame = preprocess_image(frame)
    outpath = f"runs\detect\preprocessed_frames\pp.png"
    cv2.imwrite(outpath, frame)

    results = pytesseract.image_to_string(frame)
    print(results)
    return FrameTimestamp()


def is_valid_roi(frame, roi: ROI) -> bool:
    """Return True/False depending on if an ROI contains a valid game clock with legal values for quarter and time_remaining."""

    cropped_frame = frame[roi.y1: roi.y2, roi.x1: roi.x2]
    timestamp: FrameTimestamp = extract_timestamps_from_frame(cropped_frame)
    if timestamp.quarter and timestamp.quarter:
        return True
    return False
