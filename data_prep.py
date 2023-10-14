import numpy as np
import json
import os
import csv

LOGS_DIR = r"C:/Users/Levi/Desktop/hudl-logs/DATA/game_logs"
LOGS_PATH = r"data/gamelogs_map.json"
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
        json.dump(logs, json_file, indent=4)


def get_log_path(game_id: str):
    """
    Given a game_id, ex: 974000, return the absolute path to the corrosponding game log csv.
    """

    if game_id in LOGS:
        return LOGS[game_id]
    return ""


def get_shot_events(csv_path: str, period: int):
    """
    Given a path to a game csv, return all moments at which a shot occured.
    Moment: [quarter, time_remaining].

    Discards free-throws.
    """

    shots = []

    try:
        with open(csv_path) as file:
            doc = csv.reader(file, delimiter=';')
            for row in doc:
                event = row[2]
                quarter = row[13]
                event_timestamp = row[14]
                ev_chr_1 = row[2][0]
                event_condition = ("2" == ev_chr_1 or "3" == ev_chr_1) and ("F" not in event) and (quarter == period)
                if event_condition:
                    shots.append({
                        "event": event,
                        "timestamp": float(event_timestamp)
                    })
    except:
        print(f"Error: could not open csv for path: {csv_path}")
        return []
    
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
