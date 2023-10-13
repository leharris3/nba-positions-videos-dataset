import csv
import os
import json

from data_prep import get_log_path
from shot_clip_extraction import clip_video


def get_shot_events(csv_path: str):
    """
    Given a path to a game csv, return all moments at which a shot occured.
    Moment: [quarter, time_remaining].

    Discards free-throws.
    """

    shots = []
    arr = []

    # append all rows to an array
    max_time = 0.0
    try:
        with open(csv_path) as file:
            doc = csv.reader(file, delimiter=';')
            for row in doc:
                arr.append(row)
                try:
                    max_time = max(max_time, float(row[23]))
                except:
                    pass
        period_start_time = 60.0 * (max_time // 60)
    except:
        print(f"Error: could not open csv for path: {csv_path}")
        return []

    for row in arr:
        first_char_event_str = row[2][0]
        if ("2" == first_char_event_str or "3" == first_char_event_str) and "F" not in row[2]:
            points = int(first_char_event_str)
            is_shot_made = row[2][1:]
            quarter = row[13]

            try:
                start_time = period_start_time - float(row[22])
                end_time = period_start_time - float(row[23])
                shot_start_time = round(start_time, 1)
                shot_end_time = round(end_time, 1)
                if "1" not in row[2]:
                    shots.append({
                        "attempted_points": points,
                        "result": is_shot_made,
                        "quarter": quarter,
                        "shot_start_time": shot_start_time,
                        "shot_end_time": shot_end_time
                    })
            except:
                pass

    return shots


def extract_shot_regions_from_video(video_path, shots_out_dir, timestamps_path):
    """
    Given a path to a video, a path to desired shot directory, and path to a timestamps file,
    saves a new shot clip video in a shots subdir for every shot found in a video.

    Formatting:
    Shots are saved as shot_{i}_{bool}, where bool is {true, false}. True if shot went in, false otherwise.
    """

    # mod to extract enitre shot time region and visualize results

    try:
        game_id = os.path.basename(video_path).split("_")[0]
    except:
        raise Exception(f"Bad video path: {video_path}")
    try:
        with open(timestamps_path, "r") as f:
            timestamps = json.load(f)
    except:
        print(
            f"Error: bad timestamps path: {timestamps_path}.")
        return

    path_to_logs = get_log_path(game_id)
    if path_to_logs == "":
        print(f"Error: no game logs found for video at {video_path}")
        return

    shot_events = get_shot_events(path_to_logs)
    intervals = []
    for event in shot_events:
        quarter, shot_start_time, shot_end_time = event[
            "quarter"], event["shot_start_time"], event["shot_end_time"]
        start_key = f"{quarter}_{shot_start_time}"
        end_key = f"{quarter}_{shot_end_time}"
        if start_key in timestamps and end_key in timestamps:
            try:
                start_frame = timestamps[end_key] - 10
                end_frame = timestamps[end_key] + 60
                intervals.append([start_frame, end_frame, event])
            except:
                pass
    if len(intervals) == 0:
        print(f"No shot regions found for video at path: {video_path}!")
        return

    seen_start_frames = set()
    shot_index = 0
    video_title = os.path.basename(video_path)

    for interval in intervals:
        start_frame, end_frame = interval[0], interval[1]
        if start_frame not in seen_start_frames:
            is_shot_made = interval[2]["result"]
            attempted_points = interval[2]["attempted_points"]
            output_path = os.path.join(
                shots_out_dir, f"{game_id}_clip_{shot_index}_{attempted_points}{is_shot_made}.avi")
            print(f"Shot clips saved to: {output_path}")
            clip_video(video_path, output_path, start_frame, end_frame)
            shot_index += 1
            seen_start_frames.add(start_frame)


example_game_id = "1006462"
csv_path = get_log_path(example_game_id)
res = get_shot_events(csv_path)

video_path = r"E:\Desktop\contextualized-shot-quality-estimation\testing\shot_retrevial\C-1-T\1006462_3222_Denver Nuggets_3279_New Orleans Pelicans_period2.mp4"
timestamps_path = r"testing\shot_retrevial\C-1-T-timestamps\1006462_3222_Denver Nuggets_3279_New Orleans Pelicans_period2.json"
shots_out_dir = r"testing\shot_retrevial\C-1-T-shot-clips"

extract_shot_regions_from_video(video_path, shots_out_dir, timestamps_path)
