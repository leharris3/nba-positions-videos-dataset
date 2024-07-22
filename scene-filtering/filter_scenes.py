import cv2
import os
import json
import logging
import subprocess

from glob import glob
from scenedetect import detect
from statistics import mean
from typing import List, Tuple
from scenedetect import detect, HashDetector
from scenedetect.frame_timecode import FrameTimecode
from concurrent.futures import ProcessPoolExecutor, as_completed

# TODO: where da' bugs at?!

MIN_SCENE_LEN = 2 * 30
MIN_NUM_BBXS = 6

THRESHOLD = 0.3
MIN_CONTENT_VAL = 5
FPS = 30

# setup logging and log formatting
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_scene(video_fp: str) -> List[Tuple]:
    """
    Given a path to a clip, return a list of scenes objects.
    """

    assert os.path.isfile(video_fp), f"{video_fp} does not exist"
    logger.info(f"parsing scenes from: {video_fp}")

    def append_frame_length(interval):
        start_frame, end_frame = interval
        start_frame, end_frame = int(start_frame), int(end_frame)
        length_in_frames = end_frame - start_frame
        # return the original tuple with the frame length appended
        return (start_frame, end_frame, length_in_frames)

    detector = HashDetector(threshold=THRESHOLD)
    scene_list = detect(video_fp, detector)
    logger.debug(f"scenes: {scene_list}")

    scenes_with_frames = [append_frame_length(interval) for interval in scene_list]
    logger.debug(f"scenes_with_frames: {scenes_with_frames}")

    return scenes_with_frames


def filter_scenes(video_fp: str, scenes: List[Tuple]) -> List[Tuple]:
    """
    Return scenes that are:
        1. 2+ sec. in length
        2. contain an avg. of 3+ bbxs
    """

    logger.info(f"filtering scenes from: {video_fp}")

    assert os.path.isfile(video_fp), f"{video_fp} does not exist"

    # look up annotation fp
    annotation_fp = video_fp.replace("clips", "clip-annotations").replace(
        ".mp4", "_annotation.json"
    )
    assert os.path.isfile(annotation_fp), f"{annotation_fp} does not exist"

    # load data
    with open(annotation_fp, "r") as f:
        data = json.load(f)

    # num frames to parse
    final_frame = int(cv2.VideoCapture(video_fp).get(cv2.CAP_PROP_FRAME_COUNT))
    assert final_frame > 0, f"{video_fp} has no frames"

    # count bbxs
    # [# bbxs]
    num_bbxs = [0] * final_frame
    for frame_idx in range(final_frame):
        for frame_obj in data["frames"]:
            if frame_idx == int(frame_obj["frame_id"]):
                num_bbxs[frame_idx] = len(frame_obj["bbox"])
                break

    assert len(num_bbxs) == final_frame, f"{len(num_bbxs)} != {final_frame}"

    # edge case
    if len(scenes) == 0:
        # MARK: we increase the threshold for this edge case by 2 bbxs
        if mean(num_bbxs) < (MIN_NUM_BBXS + 2):
            return []
        else:
            logger.debug(f"edge case: {mean(num_bbxs)} < {MIN_NUM_BBXS}")
            return [
                (
                    FrameTimecode(0, fps=FPS),
                    FrameTimecode(final_frame, fps=FPS),
                    final_frame,
                )
            ]

    filtered_scenes = []
    # filter scenes
    for scene in scenes:
        # 1. longer than 2s?
        scene_start = int(scene[0])
        scene_end = int(scene[1])
        logger.debug(f"scene_start: {scene_start}, scene_end: {scene_end}")

        if scene_end - scene_start < MIN_SCENE_LEN:
            logger.debug(f"scene too short: {scene_start} - {scene_end}")
            continue

        # 2. avg # bbxs < 3?
        if mean(num_bbxs[scene_start:scene_end]) < MIN_NUM_BBXS:
            logger.debug(f"not enough bbxs: {mean(num_bbxs[scene_start:scene_end])}")
            continue

        logger.debug(f"appending scene: {scene}")
        filtered_scenes.append(scene)

    logger.debug(f"filtered_scenes: {filtered_scenes}")
    return filtered_scenes


def create_new_clip(video_path: str, dst_path: str, scene) -> None:
    """
    Create a new (trimmed) clip from video_path to dst_path.
    """

    logger.info(f"creating new clip: {dst_path}")

    assert os.path.isfile(video_path), f"{video_path} does not exist"
    assert scene is not None, f"{scene} is  None"

    # save a new clip
    start_frame = int(scene[0])
    end_frame = int(scene[1])
    logger.debug(f"start_frame: {start_frame}, end_frame: {end_frame}")

    start_sec = start_frame / FPS
    end_sec = end_frame / FPS
    logger.debug(f"start_sec: {start_sec}, end_sec: {end_sec}")

    cmd = f"ffmpeg \
        -hide_banner \
        -loglevel error \
        -i {video_path} \
        -ss {start_sec} \
        -to {end_sec} \
        -c:v libx264 \
        -crf 23 \
        -preset medium \
        -c:a copy {dst_path}"

    try:
        subprocess.run(cmd, shell=True)
    except Exception as e:
        logger.error(f"Failed to create new clip: {e}")


def create_new_annotation(annotation_path: str, dst_path: str, scene):
    """
    Create a new annotation from annotation_path to dst_path.
    """

    logger.info(f"creating new annotation: {dst_path}")

    assert os.path.isfile(annotation_path), f"{annotation_path} does not exist"
    assert scene is not None, f"scene is None"

    try:
        with open(annotation_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load annotation: {e}")
        raise e

    new_annotation = {"video_id": None, "video_path": None, "frames": []}
    new_annotation["video_id"] = data["video_id"]
    new_annotation["video_path"] = data["video_path"]
    logger.debug(f"new_annotation: {new_annotation}")

    start_frame = int(scene[0])
    end_frame = int(scene[1])
    logger.debug(f"start_frame: {start_frame}, end_frame: {end_frame}")

    frames = []
    for frame in data["frames"]:
        if frame["frame_id"] >= start_frame and frame["frame_id"] <= end_frame:
            frames.append(frame)
    new_annotation["frames"] = frames
    # logger.debug(f"new_annotation: {new_annotation}")

    try:
        with open(dst_path, "w") as f:
            json.dump(new_annotation, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to create new annotation: {e}")
        raise e


def _process_clip(fp: str) -> None:
    """
    Wrapper function for segmenting a single clip.
    """

    assert os.path.isfile(fp), f"{fp} does not exist"

    logger.info(f"processing clip: {fp}")
    scenes = parse_scene(fp)
    logger.debug(f"scenes: {scenes}")

    filtered_scenes = filter_scenes(fp, scenes)
    logger.debug(f"filtered_scenes: {filtered_scenes}")

    if len(filtered_scenes) == 0:
        return

    video_dst_path = fp.replace("clips", "filtered-clips-bu")

    video_dst_dir = os.path.dirname(video_dst_path)
    os.makedirs(video_dst_dir, exist_ok=True)

    assert os.path.isdir(video_dst_dir), f"{video_dst_dir} is an invalid dir"

    logger.debug(f"video_dst_dir: {video_dst_dir}")
    logger.debug(f"video_dst_path: {video_dst_path}")

    # find annotation file
    annotation_path = fp.replace("clips", "clip-annotations").replace(
        ".mp4", "_annotation.json"
    )
    annotation_dst_path = fp.replace("clips", "filtered-clip-annotations-bu").replace(
        ".mp4", "_annotation.json"
    )
    annotation_dst_dir = os.path.dirname(annotation_dst_path)

    # recursive create dirs if doesn't exist
    os.makedirs(annotation_dst_dir, exist_ok=True)
    assert os.path.isdir(annotation_dst_dir), f"{annotation_dst_dir} was not created"

    for scene_num, scene in enumerate(filtered_scenes):
        tmp_video_dst_path = video_dst_path.replace(".mp4", f"_{scene_num}.mp4")
        tmp_annotation_dst_path = annotation_dst_path.replace(
            ".json", f"_{scene_num}.json"
        )

        logger.debug(f"creating clip with scene num {scene_num}: {tmp_video_dst_path}")
        create_new_clip(fp, tmp_video_dst_path, scene)

        logger.debug(
            f"creating annotation with scene num {scene_num}: {tmp_annotation_dst_path}"
        )
        create_new_annotation(annotation_path, tmp_annotation_dst_path, scene)


def main():

    all_clip_file_paths = glob(
        "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/clips"
        + "/*/*/*.mp4"
    )
    logger.debug(f"found {len(all_clip_file_paths)} clips")

    all_annotations_paths = glob(
        "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/clip-annotations"
        + "/*/*/*.json"
    )
    logger.debug(f"found {len(all_annotations_paths)} annotations")

    all_annotation_basenames = set(
        list(
            os.path.basename(fp).replace(".json", ".mp4").replace("_annotation", "")
            for fp in all_annotations_paths
        )
    )
    clips_w_ann_file_paths = [
        fp
        for fp in all_clip_file_paths
        if os.path.basename(fp) in all_annotation_basenames
    ]

    with ProcessPoolExecutor(max_workers=64) as pool:
        pool.map(_process_clip, clips_w_ann_file_paths)


if __name__ == "__main__":
    main()
