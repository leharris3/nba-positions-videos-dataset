import cv2
import os
import json

from tqdm import tqdm
from glob import glob
from scenedetect import detect
from statistics import mean
from typing import List, Tuple
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from concurrent.futures import ProcessPoolExecutor, as_completed


MIN_SCENE_LEN = 2 * 30
MIN_NUM_BBXS = 3
THRESHOLD = 30
FPS = 30

detector = ContentDetector(threshold=THRESHOLD)


def parse_scene(video_fp: str) -> List[Tuple]:

    def add_frame_length_info(interval):
        def parse_frame_info(frame_info):
            return int(frame_info)

        def calculate_length_in_frames(start_info, end_info):
            start_frame = parse_frame_info(start_info)
            end_frame = parse_frame_info(end_info)
            return end_frame - start_frame

        start_info, end_info = interval
        length_in_frames = calculate_length_in_frames(start_info, end_info)
        # Return the original tuple with the frame length appended
        return (start_info, end_info, length_in_frames)

    print(f"parsing {video_fp}")
    scene_list = detect(video_fp, detector)
    return [add_frame_length_info(interval) for interval in scene_list]


def filter_scenes(video_fp: str, scenes: List[Tuple]) -> List[Tuple]:
    """
    Return scenes that are:
        1. 2+ sec. in length
        2. contain an avg. of 3+ bbxs
    """

    # look up annotation fp
    annotation_fp = video_fp.replace("clips", "clip-annotations").replace(
        ".mp4", "_annotation.json"
    )
    # load data
    with open(annotation_fp, "r") as f:
        data = json.load(f)

    # find all frames w/ bbxs
    unique_keys = set()
    for item in data["frames"]:
        frame_id = item["frame_id"]
        if frame_id not in unique_keys:
            unique_keys.add(int(frame_id))

    # num frames to parse
    final_frame = int(cv2.VideoCapture(video_fp).get(cv2.CAP_PROP_FRAME_COUNT))
    # count bbxs
    # [# bbxs]
    num_bbxs = []
    for frame_idx in range(final_frame):
        if frame_idx not in unique_keys:
            num_bbxs.append(0)
        else:
            num_bbx_tmp = len(data["frames"][frame_idx]["bbox"])
            num_bbxs.append(num_bbx_tmp)

    # edge case
    if len(scenes) == 0:
        if mean(num_bbxs) < MIN_NUM_BBXS:
            return []
        else:
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
        scene_start = scene[0].frame_num
        scene_end = scene[1].frame_num
        if scene_end - scene_start < MIN_SCENE_LEN:
            continue
        # 2. avg # bbxs < 3?
        if mean(num_bbxs[scene_start:scene_end]) < MIN_NUM_BBXS:
            continue
        filtered_scenes.append(scene)

    return filtered_scenes


def create_new_clip(video_path: str, dst_path: str, scene):
    # save a new clip
    start_frame = scene[0].frame_num
    end_frame = scene[1].frame_num
    start_sec = start_frame / FPS
    end_sec = end_frame / FPS
    cmd = f"ffmpeg -hide_banner -loglevel error -i {video_path} -ss {start_sec} -to {end_sec} -c:v libx264 -crf 23 -preset medium -c:a copy {dst_path}"
    os.system(cmd)


def create_new_annotation(annotation_path: str, dst_path: str, scene):
    with open(annotation_path, "r") as f:
        data = json.load(f)

    new_annotation = {"video_id": None, "video_path": None, "frames": []}
    new_annotation["video_id"] = data["video_id"]
    new_annotation["video_path"] = data["video_path"]

    start_frame = scene[0].frame_num
    end_frame = scene[1].frame_num
    frames = []
    for frame in data["frames"]:
        if frame["frame_id"] >= start_frame and frame["frame_id"] <= end_frame:
            frames.append(frame)
    new_annotation["frames"] = frames

    with open(dst_path, "w") as f:
        json.dump(new_annotation, f, indent=4)


def process_clip(fp):
    scenes = parse_scene(fp)
    filtered_scenes = filter_scenes(fp, scenes)

    video_dst_path = fp.replace("clips", "filtered-clips")
    video_dst_dir = os.path.dirname(video_dst_path)
    print(f"video_dst_dir: {video_dst_dir}")

    # find annotation file
    annotation_path = fp.replace("clips", "clip-annotations").replace(
        ".mp4", "_annotation.json"
    )
    annotation_dst_path = fp.replace("clips", "filtered-clip-annotations").replace(
        ".mp4", "_annotation.json"
    )
    annotation_dst_dir = os.path.dirname(annotation_dst_path)

    # recursive create dirs if doesn't exist
    os.makedirs(video_dst_dir, exist_ok=True)
    os.makedirs(annotation_dst_dir, exist_ok=True)

    for scene_num, scene in enumerate(filtered_scenes):
        video_dst_path = video_dst_path.replace(".mp4", f"_{scene_num}.mp4")
        annotation_dst_path = annotation_dst_path.replace(".json", f"_{scene_num}.json")
        create_new_clip(fp, video_dst_path, scene)
        create_new_annotation(annotation_path, annotation_dst_path, scene)
        print(f"created {video_dst_path} and \n{annotation_dst_path}")


def main():

    all_clip_file_paths = glob(
        "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/clips"
        + "/*/*/*.mp4"
    )
    print(f"found {len(all_clip_file_paths)} clips")

    all_annotations_paths = glob(
        "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/clip-annotations"
        + "/*/*/*.json"
    )
    print(f"found {len(all_annotations_paths)} annotations")

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

    # create a progress bar object
    progress_bar = tqdm(total=len(clips_w_ann_file_paths))

    with ProcessPoolExecutor(max_workers=64) as executor:
        executor.map(process_clip, clips_w_ann_file_paths)
        # update progress bar
        for _ in as_completed(executor):
            progress_bar.update(1)


if __name__ == "__main__":
    main()
