# video_processor.py

import yaml
import argparse
import os
import logging
import json
import shutil
import av
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from utils._models import YOLOModel, Florence
from utils.extract_roi import extract_roi_from_video
from utils.extract_time_remaining import ocr
from utils._grab_frames import video_to_frames
from utils._file_helpers import NumpyEncoder

logger = logging.getLogger(__name__)


def extract_timestamps_from_video(
    rank,
    config,
    video_path: Path,
    yolo_model=None,
    florence_model=None,
    florence_processor=None,
) -> Optional[Dict]:
    """
    Extract timestamps from a single video.

    Args:
        config (Dict): Configuration dictionary.
        video_path (Path): Path to the video file.

    Returns:
        Optional[Dict]: Extracted timestamps or None if already processed.
    """

    def save_frames(
        video_path: Path, temp_dir: Path, bbox: List[int], step: int
    ) -> None:
        """
        Save frames from a video to a temporary directory.

        Args:
            video_path (Path): Path to the video file.
            temp_dir (Path): Path to the temporary directory.
            bbox (List[int]): ROI coordinates [x1, y1, x2, y2].
            step (int): Frame step for saving.
        """
        video_to_frames(video_path, temp_dir, bbox, every=step)

    output_dir = Path(config["output_dir"])
    timestamp_out_path = Path(output_dir / f"{video_path.name.replace('mp4', '')}")

    if not video_path.exists():
        raise FileNotFoundError(f"Error: Video file not found: {video_path}")
    if timestamp_out_path.is_file():
        logging.info(f"Skipping already processed video: {video_path}")
        return None

    # extract the bbox around the game clock
    logger.info(f"Extracting ROI from video: {video_path}")
    time_remaining_roi = extract_roi_from_video(config, video_path, yolo_model)
    if time_remaining_roi is None:
        logging.warning(f"Could not extract ROI from video: {video_path}")
        return None
    # bbox format: x1, y1, x2, y2
    bbox = time_remaining_roi.tolist()

    # create tmp dir to save tracklet frames to
    logger.info(f"Extracting frames from: {video_path}")
    temp_dir = Path(config["temp_frames_dir"]) / video_path.name.replace(".mp4", "")
    temp_dir.mkdir(exist_ok=True)
    save_frames(video_path, temp_dir, bbox, config["time_remaining_step"])

    # get all images paths in the tmp dir
    logger.info(f"Extracting time remaining from: {video_path}")
    image_paths = list(temp_dir.glob("*.jpg"))
    results = ocr(rank, config, image_paths, florence_model, florence_processor)

    # clean up temporary directory
    shutil.rmtree(temp_dir)
    return results


def process_directory(rank: int, config) -> None:
    """
    Process all videos in a directory and extract timestamps.

    Args:
        rank (int): Current gpu device id.
        config (Dict): Configuration dictionary.
    """

    # for now we'll assume we only use unaltered basenames in temp and results dir
    def get_remaining_file_paths():
        all_vids_dir = config["input_dir"]
        output_dir = config["output_dir"]
        temp_frames_dir = config["temp_frames_dir"]
        # get all processed / in-progress videos
        results_basenames = set(
            list(
                os.path.basename(fp).replace(".json", "")
                for fp in glob(output_dir + "/*.json")
            )
        )
        tmp_dir_basenames = set(
            list(os.path.basename(fp) for fp in glob(temp_frames_dir + "/*"))
        )
        
        processed_files = results_basenames | tmp_dir_basenames
        remaining_file_paths = []
        for fp in glob(all_vids_dir + "/*.mp4"):
            basename = os.path.basename(fp).replace(".mp4", "")
            if basename not in processed_files:
                remaining_file_paths.append(fp)
        return remaining_file_paths

    yolo_model = YOLOModel.get_model(config).to(rank)
    florence_model, florence_processor = Florence.get_model_and_processor(config)

    # TODO: might run into errors when using mix-precision and copying to GPU
    florence_model = florence_model.to(rank)

    out_dir = Path(config["output_dir"])
    while True:
        remaining_file_paths = get_remaining_file_paths()
        # all done!
        if len(remaining_file_paths) == 0:
            break
        next_fp = Path(remaining_file_paths[0])
        logger.info(f"Extracting timestamps from video: {next_fp}")
        results = extract_timestamps_from_video(
            rank, config, next_fp, yolo_model, florence_model, florence_processor
        )
        if results:
            output_file = out_dir / f"{next_fp.name.replace('.mp4', '')}.json"
            with output_file.open("w") as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)


def main() -> None:
    """
    Main function to run the video processing script.
    """

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
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    setup_logging(config["log_level"])
    logging.info(f"Starting video processing with input: {config['input_dir']}")
    
    # free any memory we can
    torch.cuda.empty_cache()
    
    # spawn process dir jobs
    mp.spawn(
        process_directory,
        args=([config]),
        nprocs=torch.cuda.device_count(),
        join=True,
    )
    logging.info("Script execution completed")


if __name__ == "__main__":
    main()
