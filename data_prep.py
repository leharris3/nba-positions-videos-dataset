import numpy as np
import json
import os
import csv

LOGS_PATH = r"data\gamelogs_map.json"
with open(LOGS_PATH, 'r') as logs_map:
    LOGS = json.load(logs_map)


def generate_game_log_map(save_logs_to_path: str, logs_dir: str):
    """
    Given a path to a logs dir, output a .json file containing game ids and their paths.
    """

    logs = {}
    for root, _, files in os.walk(logs_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                log_id = file.split("_")[0]
                logs[log_id] = csv_file_path
    with open(save_logs_to_path, 'w') as json_file:
        json.dump(logs, json_file)


def get_log_path(game_id: str):
    """
    Given a game_id, ex: 974000, return the absolute path to the corrosponding game log csv.
    """

    if game_id in LOGS:
        return LOGS[game_id]
    return ""


def get_shot_events(csv_path: str):
    """
    Given a path to a game csv, return all moments at which a shot occured.
    Moment: [quarter, time_remaining]
    """

    shots = []
    arr = []
    # append all rows to an array
    max_time = 0.0
    with open(csv_path) as file:
        doc = csv.reader(file, delimiter=';')
        for row in doc:
            arr.append(row)
            try:
                max_time = max(max_time, float(row[23]))
            except:
                pass

    period_start_time = 60.0 * (max_time // 60)
    print("Max Time", period_start_time)
    for row in arr:
        if "+" in row[2] or "-" in row[2]:
            shot = row[2]
            quarter = row[13]
            start_time = round(period_start_time - float(row[22]), 1)
            end_time = round(period_start_time - float(row[23]), 1)
            shots.append([shot, quarter, start_time, end_time])
    return shots


def get_shot_time_intervals(path_to_timestamps: str, shot_events):
    """
    Given a path to a game's timestamps and array containing shot events,
    return a list of shot events with frame intervals.

    Out: [shot_type, start_frame, end_frame, [quarter, start_time, end_time]]...
    """

    time_intervals = []
    with open(path_to_timestamps, "r") as raw_json:
        timestamp_json = json.load(raw_json)

    # TODO: Update save timestamps method to use form:
    # "quarter_time_remaining: frame"

    for frame_index in timestamp_json:
        timestamp = timestamp_json[frame_index]
        print(timestamp)

    pass


game_id = "1003170"
csv_path = r"data\1003170_Washington Wizards - Portland Trail Blazers.csv"
shots = get_shot_events(csv_path)
print(shots)

# get_shot_time_intervals(
#     r"data\940432_2900_KK MZT Skopje Aerodrom_2953_KK Buducnost_period2.json",
#     [])
