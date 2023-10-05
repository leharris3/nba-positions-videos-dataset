import os
import json

dir_path = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/400"
vids = os.listdir(dir_path)

with open("gamelogs_map.json", "r") as json_file:
    data = json.load(json_file)

# for some reason we only seem to have logs for ~1/4 video clips!
found, total = 0, 0
for vid in vids:
    prefix = vid[0:6]
    if prefix in data:
        found += 1
    total += 1
print(found, "/", total)

# TODO: new algo:
# for vid in dir
    # if vid has log:
        # extract_shot_clips
