import random
import os
import json
import shutil
from temporal_grounding import *

with open("shot-clip-extraction/gamelogs_map.json") as f:
    LOGS = json.load(f)
HUDL_GAMES_DIR = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/400"
BATCH_DIR = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/videos_with_logs_batch_100"


def process_video():
    pass


def get_all_valid_game_paths():

    hudl_games_dir = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/400"
    vids = os.listdir(hudl_games_dir)
    valid_game_paths = []
    for vid in vids:
        game_id = vid[0: 6]
        vid_path = os.path.join(HUDL_GAMES_DIR, vid)
        if game_id in LOGS:
            valid_game_paths.append(vid_path)
    print(len(valid_game_paths), "/", len(vids))
    return valid_game_paths


def move_100_videos_to_batch_dir(batch_dir=None):

    batch_dir = BATCH_DIR
    batch_size = 1
    paths = get_all_valid_game_paths()[0:batch_size]

    for old_path in paths:
        name = os.path.basename(old_path)
        new_path = os.path.join(batch_dir, name)
        shutil.move(old_path, new_path)


def process_dir(dir_path):
    pass


def extract_shots_from_video(video_path, timestamps_path, gamelog_path, to_path):
    pass


def fetch_first_shot(video_path: str, game_id: str, timestamps_path: str):
    if game_id not in LOGS:
        raise Exception(
            f"Error: game with id {game_id} does not have a valid game log.")
    else:
        return "found"


def choose_ten_games_with_logs():
    vids = os.listdir(HUDL_GAMES_DIR)
    random.shuffle(vids)

    move_path = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/C-10-L"
    moved = 0
    for vid in vids:
        id = vid[0: 6]
        vp = os.path.join(HUDL_GAMES_DIR, vid)
        mp = os.path.join(move_path, vid)
        if id in LOGS:
            shutil.copy(vp, mp)
            moved += 1
        if moved >= 10:
            break
