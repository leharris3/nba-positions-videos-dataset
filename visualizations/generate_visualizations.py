import os
from viz import *
from datascience import *

videos_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/game-replays/720"
timestamps_dir = r"/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamps/post-processed"
viz_dir = r'/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamp-visualizations'

timestamps_paths = {}
game_ids = []

timestamps = os.listdir(timestamps_dir)
for t in timestamps:
    if not '.json' in t:
        continue
    fp = os.path.join(timestamps_dir, t)
    period = t.split('_period')[1][0]
    game_id = t.split('_')[0] + '_' + period
    timestamps_paths[game_id] = fp

videos = os.listdir(videos_dir)
video_paths = []
timestamp_paths = []

for vid in videos:
    if not '.mp4' in vid:
        continue
    fp = os.path.join(videos_dir, vid)
    period = vid.split('_period')[1][0]
    game_id = vid.split('_')[0] + '_' + period
    if game_id in timestamps_paths:
        timestamp_paths.append(timestamps_paths[game_id])
    else:
        timestamp_paths.append('')
    game_ids.append(game_id)
    video_paths.append(fp)

# table of 100 random timestamp video pairs
table = Table().with_columns(
    'game_id', game_ids,
    'video_path', video_paths,
    'timestamps_path', timestamp_paths
)
table.where('timestamps_path', are.not_equal_to('')).sort('game_id').shuffle().take(100)\

# generate some viz
for row in table.rows:
    try:
        video_path = row[1]
        timestamp_path = row[2]
        game_id = row[0]
        out_path = os.path.join(viz_dir, game_id + '.mp4')
        if os.path.exists(out_path):
            continue

        visualize_timestamps(
            video_path,
            timestamp_path,
            out_path
        )
    except:
        pass 