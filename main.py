import os
import time
from temporal_grounding import *
from visualizations.viz import *
import argparse

vids_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/game-replays/720"
timestamps_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamps"
viz_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamp-visualizations"
preprocessed_videos = r"processed_vids.txt"

process_dir(
    dir_path=vids_dir,
    data_out_path=timestamps_dir,
    viz_out_path=None,
    preprocessed_videos=preprocessed_videos,
)