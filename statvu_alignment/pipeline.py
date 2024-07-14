# video_processor.py

import yaml
import argparse
import os
import logging
import concurrent.futures
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import cv2
from tqdm import tqdm

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils._models import YOLOModel
from utils.extract_roi import extract_roi_from_video
from utils.extract_time_remaining import FlorenceModel, ocr

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """
    Set up logging configuration.

    Args:
        log_level (str): Desired logging level.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="video_processor.log",
    )


def process_directory(config: Dict) -> None:
    """
    Process all videos in a directory and extract timestamps.

    Args:
        config (Dict): Configuration dictionary.
    """
    src_dir = Path(config["input_dir"])
    out_dir = Path(config["output_dir"])
    if not src_dir.is_dir():
        raise ValueError(f"Error: Invalid video directory: {src_dir}")
    video_file_paths = list(src_dir.glob("*.mp4"))
    for video_path in video_file_paths:
        logger.info(f"Processing video: {video_path}")
        results = extract_timestamps_from_video(config, video_path)
        if results:
            output_file = out_dir / f"{video_path.name}.json"
            with output_file.open("w") as f:
                json.dump(results, f)


def extract_timestamps_from_video(config: Dict, video_path: Path) -> Optional[Dict]:
    """
    Extract timestamps from a single video.

    Args:
        config (Dict): Configuration dictionary.
        video_path (Path): Path to the video file.

    Returns:
        Optional[Dict]: Extracted timestamps or None if already processed.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Error: Video file not found: {video_path}")
    output_dir = Path(config["output_dir"])
    timestamp_out_path = output_dir / f"{video_path.name}.json"
    if timestamp_out_path.is_file():
        logging.info(f"Skipping already processed video: {video_path}")
        return None
    yolo_model = YOLOModel.get_model(config["device"])
    time_remaining_roi = extract_roi_from_video(
        video_path, yolo_model, device=config["device"]
    )
    if time_remaining_roi is None:
        logging.warning(f"Could not extract ROI from video: {video_path}")
        return None
    tr_x1, tr_y1, tr_x2, tr_y2 = time_remaining_roi
    temp_dir = Path(f"temp_{video_path.stem}")
    temp_dir.mkdir(exist_ok=True)
    save_frames(
        video_path, temp_dir, tr_x1, tr_y1, tr_x2, tr_y2, config["time_remaining_step"]
    )
    florence_model, processor = FlorenceModel.load_model_and_tokenizer(config)
    image_paths = list(temp_dir.glob("*.png"))
    results = ocr(config, image_paths, florence_model, processor)
    # Clean up temporary directory
    for file in temp_dir.glob("*"):
        file.unlink()
    temp_dir.rmdir()
    return results


def save_frames(
    video_path: Path, temp_dir: Path, x1: int, y1: int, x2: int, y2: int, step: int
) -> None:
    """
    Save frames from a video to a temporary directory.

    Args:
        video_path (Path): Path to the video file.
        temp_dir (Path): Path to the temporary directory.
        x1, y1, x2, y2 (int): ROI coordinates.
        step (int): Frame step for saving.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame_number in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[y1:y2, x1:x2]
            frame_filename = temp_dir / f"{frame_number:05d}.png"
            futures.append(executor.submit(cv2.imwrite, str(frame_filename), frame))
        concurrent.futures.wait(futures)
    cap.release()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main() -> None:
    """
    Main function to run the video processing script.
    """
    parser = argparse.ArgumentParser(
        description="Process videos and extract timestamps."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config["log_level"])
    logging.info(f"Starting video processing with input: {config['input_dir']}")
    try:
        process_directory(config)
    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
    logging.info("Script execution completed")


if __name__ == "__main__":
    main()
