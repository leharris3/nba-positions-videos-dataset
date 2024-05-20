import os
import json
import tqdm
import signal
from matplotlib import pyplot as plt
import numpy as np
from datascience import *
import matplotlib
matplotlib.use('TkAgg')
from post_processing import post_process_timestamps

timestamps_dir = '/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamps'
dst_dir = '/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/timestamps/post-processed'
timestamps = os.listdir(timestamps_dir)

# post process timestamps directory
# why are we getting suck on some files?

def handler(signum, frame):
    raise Exception("Processing Timeout")

signal.signal(signal.SIGALRM, handler)
for ts in tqdm.tqdm(timestamps):
    try:
        # Skip non-JSON files
        if not ts.endswith('.json'):
            continue

        fp = os.path.join(timestamps_dir, ts)
        dst = os.path.join(dst_dir, ts) 

        # Set alarm for 5 seconds
        signal.alarm(60)

        with open(fp, 'r') as f:
            timestamp = json.load(f)

        found = any(v['time_remaining'] is not None for v in timestamp.values())
        if not found:
            continue

        timestamp = post_process_timestamps(timestamp)

        with open(dst, 'w') as f:
            json.dump(timestamp, f, indent=4)

        # Reset alarm
        signal.alarm(0)

    except Exception as e:
        print(f"Error processing {ts}: {e}")

    finally:
        # Disable the alarm
        signal.alarm(0)
