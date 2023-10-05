# Scripts for fetching / saving shot clips from videos given timestamps and game logs.

import os
import json

# want to: given a path to a video and timestamps: produce viz of general time region containing a shot


class Gamelogs:

    with open("shot-clip-extraction/gamelogs_map.json") as f:
        logs = json.load(f)

    hudl_games_dir = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/400"


def fetch_first_shot(video_path: str, game_id: str, timestamps_path: str):
    if game_id not in Gamelogs.logs:
        raise Exception(
            f"Error: game with id {game_id} does not have a valid game log.")
    else:
        return "found"


def find_first_game_with_log():

    vids = os.listdir(Gamelogs.hudl_games_dir)
    valid_game_ids = []
    for vid in vids:
        game_id = vid[0: 6]
        if game_id in Gamelogs.logs:
            path_to_csv = Gamelogs.logs[game_id]
            return path_to_csv
