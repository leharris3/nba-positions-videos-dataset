import random
import os
import json
import shutil
import cv2
import temporal_grounding.temporal_grounding as tg
from temporal_grounding.temporal_grounding import *
from data_prep import *


def process_dir(videos_dir, timestamps_dir, shots_dir, extract_timestamps=False):
    """
    Extract shot from every video in a dir to a specifed 'shots dir'.
    Option to extract timestamps from videos before extracting shots.
    """
    if extract_timestamps:
        tg.process_dir(videos_dir, timestamps_dir)
    vids = os.listdir(videos_dir)
    for vid in vids:
        video_path = os.path.join(videos_dir, vid)
        timestamps_path = os.path.join(
            timestamps_dir, vid.replace(".avi", ".json"))
        extract_shots_from_video(video_path, shots_dir, timestamps_path)


def extract_shots_from_video(video_path, shots_dir, timestamps_path):
    """
    Given a path to a video, a path to desired shot directory, and path to a timestamps file,
    saves a new shot clip video in a shots subdir for every shot found in a video.

    Formatting:
    Shots are saved as shot_{i}_{bool}, where bool is {true, false}. True if shot went in, false otherwise.
    """

    try:
        game_id = os.path.basename(video_path).split("_")[0]
    except:
        raise Exception(f"Bad video path: {video_path}")
    try:
        with open(timestamps_path, "r") as f:
            timestamps = json.load(f)
    except:
        print(
            f"Error: bad timestamps path: {timestamps_path}.")
        return

    path_to_logs = get_log_path(game_id)
    if path_to_logs == "":
        print(f"Error: no game logs found for video at {video_path}")
        return

    shot_events = get_shot_events(path_to_logs)
    intervals = []
    for event in shot_events:
        key = f"{event[2]}_{event[3]}"
        if key in timestamps:
            try:
                backwards_pad = 30 * 5
                shot_duration = 30 * 10
                start_frame = timestamps[key] - backwards_pad
                end_frame = start_frame + shot_duration
                intervals.append([start_frame, end_frame, event])
            except:
                pass
    if len(intervals) == 0:
        return

    print(intervals)
    see_start_frames = set()

    shot_index = 0
    video_title = os.path.basename(video_path)
    shot_subdir = os.path.join(shots_dir, video_title.replace(".avi", ""))
    if len(intervals) > 0:
        os.makedirs(shot_subdir, exist_ok=True)
    for interval in intervals:
        start_frame, end_frame = interval[0], interval[1]
        if start_frame not in see_start_frames:
            is_shot_made = interval[2][1]
            output_path = os.path.join(
                shot_subdir, f"clip_{shot_index}_{is_shot_made}.avi")
            print(f"Shot clips saved to: {output_path}")
            clip_video(video_path, output_path, start_frame, end_frame)
            shot_index += 1
            see_start_frames.add(start_frame)


def clip_video(input_path, output_path, start_frame, end_frame):
    """
    Uhh... what do you think this function does????
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
                break
        cap.release()
        out.release()
    except Exception as e:
        print(f"Error: {str(e)}")
