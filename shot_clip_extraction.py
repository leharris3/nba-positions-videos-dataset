import random
import os
import json
import shutil
import cv2
from data_prep import *


def process_dir(videos_dir, shots_dir):
    """
    Extract shot from every video in a dir to a specifed 'shots dir'.
    Option to extract timestamps from videos before extracting shots.
    """

    vids = os.listdir(videos_dir)
    for vid in vids:
        video_path = os.path.join(videos_dir, vid)
        extract_shots_from_video(video_path, shots_dir)


def extract_shots_from_video(path_to_video, clips_dir):
    """
    Given a path to a video, a path to desired shot directory, and path to a timestamps file,
    saves a new shot clip video in a shots subdir for every shot found in a video.

    Formatting:
    Shots are saved as shot_{i}_{bool}, where bool is {true, false}. True if shot went in, false otherwise.
    """

    try:
        game_id = os.path.basename(path_to_video).split("_")[0]
        period = path_to_video[-5]
    except:
        raise Exception(f"Bad video path: {path_to_video}")
    path_to_logs = get_log_path(game_id)
    if path_to_logs == "":
        print(f"Error: no game logs found for video at {path_to_video}")
        return
    
    cap = cv2.VideoCapture(path_to_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    shot_events = get_shot_events(path_to_logs, period)
    for shot in shot_events:
        event = shot["event"]
        timestamp = float(shot["timestamp"])
        framestamp = int(fps * timestamp)
        start_frame = int(framestamp - (fps * 2))
        end_frame = int(framestamp + (fps * 2))
        out_path = os.path.join(clips_dir, f"{event}_{game_id}_{timestamp}.mp4")
        clip_video(path_to_video, out_path, start_frame, end_frame)
        


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
