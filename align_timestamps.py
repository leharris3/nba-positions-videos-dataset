import os
import json
import yaml

from datascience import *
from post_processing.post_processing import *

timestamps_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamps/post-processed"
statvu_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/statvu-game-logs/"
out_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/2d-player-positions"

statvu_paths = [os.path.join(statvu_dir, f) for f in os.listdir(statvu_dir)]
timestamps_paths = [os.path.join(timestamps_dir, f) for f in os.listdir(timestamps_dir)]
statvu_matched_paths = []  # paths to all statvu files with a matching timestamp file

# find all statvu files with a matching timestamp path
found = False
for fp in timestamps_paths:
    id = fp.split("/")[12].split("_")[0]
    for sv_path in statvu_paths:
        if id in sv_path:
            statvu_matched_paths.append(sv_path)
            found = True
            break
    if not found:
        statvu_matched_paths.append("")
    found = False

path_map = (
    Table()
    .with_columns(
        "timestamp_path", timestamps_paths, "statvu_dir_path", statvu_matched_paths
    )
    .where("statvu_dir_path", are.not_equal_to(""))
)

print(f"Processing {path_map.num_rows} files.")
print(f"Eta: {20 * path_map.num_rows} seconds!")

# generate mapped player position files to outdir
for row in path_map.rows:
    timestamp_path = row[0]
    statvu_dir = row[1]
    statvu_path = ""
    # try:
    # kind of dumb, each stavu file is kept in a one file dir, idk

    try:
        for f in os.listdir(statvu_dir):
            f = f.decode()
            if ".json" in f:
                statvu_path = os.path.join(statvu_dir, f)
                break
        game_id = statvu_dir.split("/")[-1]
        quarter = timestamp_path.split("/")[-1].split("_period")[-1][0]
        new_name = game_id + "." + "Q" + quarter + "." + "2D-POS" + ".json"
        new_path = os.path.join(out_dir, new_name)
        moments = get_unique_moments_from_statvu(
            statvu_path
        )  # load in all raw player positions
        # open timestamps
        with open(timestamp_path, "r") as f:
            timestamp_data = json.load(f)
        # generate a player positions file
        mapped_data = map_frames_to_moments(
            timestamp_data,
            moments,
        )
        # save player positions file
        with open(new_path, "w") as f:
            json.dump(mapped_data, f)
    except:
        print(f"error: could not process video at: {timestamp_path}")
