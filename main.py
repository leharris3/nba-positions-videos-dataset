import os
import time
from temporal_grounding import *
from viz import *

# vids_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/My Mac/720"
# timestamps_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamps"
# viz_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamp-visualizations"
# preprocessed_videos = r"processed_vids.txt"

# process_dir(
#     dir_path=vids_dir,
#     data_out_path=timestamps_dir,
#     viz_out_path=viz_dir,
#     preprocessed_videos=preprocessed_videos
# )

video_path = 'demo/video_dir/707_01-17-2016_2976_Memphis Grizzlies_3173_New York Knicks_period1.mp4'
process_video(
    video_title='707_01-17-2016_2976_Memphis Grizzlies_3173_New York Knicks_period1.mp4',
    video_dir='demo/video_dir',
    data_dir='demo/timestamps_dir',
    viz_dir='demo/viz_dir'
)