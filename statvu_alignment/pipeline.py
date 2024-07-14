# video_processor.py

import yaml
import argparse
import os
import logging
import json
import av
import numpy as np

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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
        logger.info(f"Extracting timestamps from video: {video_path}")
        results = extract_timestamps_from_video(config, video_path)
        if results:
            output_file = out_dir / f"{video_path.name}.json"
            with output_file.open("w") as f:
                json.dump(results, f)
        # MARK: BREAK
        break


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
    timestamp_out_path = Path(output_dir / f"{video_path.name}.json")
    if timestamp_out_path.is_file():
        logging.info(f"Skipping already processed video: {video_path}")
        return None
    logger.info("Loading YOLO model...")
    yolo_model = YOLOModel.get_model(config)
    logger.info(f"Extracting roi from video: {video_path}")
    time_remaining_roi = extract_roi_from_video(config, video_path, yolo_model)
    if time_remaining_roi is None:
        logging.warning(f"Could not extract ROI from video: {video_path}")
        return None
    # bbox format: x1, y1, x2, y2
    bbox = time_remaining_roi.tolist()
    # create tmp dir to save tracklet frames to
    temp_dir = Path(config["temp_frames_dir"]) / video_path.name
    temp_dir.mkdir(exist_ok=True)
    logger.info(f"Extracting frames from: {video_path}")
    save_frames(video_path, temp_dir, bbox, config["time_remaining_step"])
    florence_model, processor = FlorenceModel.load_model_and_tokenizer(config)
    image_paths = list(temp_dir.glob("*.png"))
    logger.info(f"Extracting time remaining from: {video_path}")
    results = ocr(config, image_paths, florence_model, processor)
    # clean up temporary directory
    for file in temp_dir.glob("*"):
        file.unlink()
    temp_dir.rmdir()
    return results


# TODO: unacceptably slow
def process_frame(frame_data):
    frame_number, frame, bbox, output_path = frame_data
    x1, y1, x2, y2 = bbox
    frame = frame[y1:y2, x1:x2]
    output_file = output_path / f"{frame_number:05d}.jpg"
    frame.tofile(str(output_file))


def save_frames(video_path: Path, temp_dir: Path, bbox: List[int], step: int) -> None:
    """
    Save frames from a video to a temporary directory.

    Args:
        video_path (Path): Path to the video file.
        temp_dir (Path): Path to the temporary directory.
        bbox (List[int]): ROI coordinates [x1, y1, x2, y2].
        step (int): Frame step for saving.
    """
    with av.open(str(video_path)) as container:
        video = next(s for s in container.streams if s.type == "video")
        total_frames = video.frames
        frames_to_process = []
        for frame_number, frame in enumerate(
            tqdm(container.decode(video), total=total_frames, desc="Extracting frames")
        ):
            if frame_number % step == 0:
                np_frame = np.array(frame.to_image())
                frames_to_process.append((frame_number, np_frame, bbox, temp_dir))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(
            tqdm(
                executor.map(process_frame, frames_to_process),
                total=len(frames_to_process),
                desc="Saving frames",
            )
        )


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
    process_directory(config)
    logging.info("Script execution completed")


if __name__ == "__main__":
    main()
