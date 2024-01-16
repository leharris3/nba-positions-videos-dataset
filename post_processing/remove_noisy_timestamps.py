import os
import json
from matplotlib import pyplot as plt
import numpy as np
from datascience import *
import tqdm

def remove_noisey_timestamps(timestamps_dir: str):

    def get_time_remaing_array_from_fp(fp):
        try:
            arr = []
            with open(fp, 'r') as f:
                data = json.load(f)
            for k in data:
                tr = data[k]['time_remaining']
                arr.append(float(tr))
            return np.array(arr)
        except:
            return []
        pass

    # get all timestamp paths
    timestamps_paths = [os.path.join(timestamps_dir, f) for f in os.listdir(timestamps_dir)]

    # calculate noise scores for each timestamp file and put it into a big table
    timestamp_avg_noise = []
    timestamp_paths_trimmed = []
    for ts in tqdm.tqdm(timestamps_paths):
        if '.json' not in ts:
            continue
        time_remaining_arr = get_time_remaing_array_from_fp(ts)
        expected_arr = np.arange(0, 720, 720 / len(time_remaining_arr))[::-1]
        time_remaining_arr.resize(expected_arr.shape)
        diff = time_remaining_arr - expected_arr
        avg_noise = np.average(diff)
        timestamp_avg_noise.append(avg_noise)
        timestamp_paths_trimmed.append(ts)

    # select only timestamps with high noise scores
    timestamps: Table = Table().with_columns(
        'path', timestamp_paths_trimmed,
        'avg_noise', timestamp_avg_noise,
        'in_range', np.less(timestamp_avg_noise, 50) & np.greater(timestamp_avg_noise, -75)
    ).where('in_range', are.equal_to(False)).drop('in_range')

    # remove all noisy timestamps
    noisy_paths = timestamps.column('path')
    print(f'About to remove {len(noisy_paths)} files!')

    for fp in noisy_paths:
        print(f'Removing file at {fp}!')
        os.remove(fp)

dir_path = r'/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamps/post-processed'
remove_noisey_timestamps(dir_path)
