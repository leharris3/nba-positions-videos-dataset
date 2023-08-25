import sys
import json
from typing import Dict, Any, List, Union

NO_MATCH = -1


def print_progress(progress: float) -> None:
    progress_bar = "[" + "#" * \
        int(progress * 20) + " " * (20 - int(progress * 20)) + "]"
    sys.stdout.write("\r{} {:.2f}%".format(progress_bar, progress * 100))
    sys.stdout.flush()


def map_timestamps_to_statvu(video_path: str, timestaps_path: str, out_path: str) -> None:
    """Maps a frame parameter to every moment in orginal stavu data file."""

    timestamps = {}
    timestamp_data_raw = json.load(open(timestaps_path))
    for frame in timestamp_data_raw:
        temp_quarter = str(timestamp_data_raw[frame][0])
        temp_time_remaining = str(timestamp_data_raw[frame][1])
        key = f"{temp_quarter} {temp_time_remaining}"
        timestamps[key] = frame

    statvu_data: Dict[str, Any] = json.load(
        open(game.data.path_to_processed_data_unzipped))
    statvu_data["video_path"] = game.data.path_to_processed_data_unzipped.strip(
        '.json') + '.mp4'

    total, misses = 0, 0
    for event in statvu_data['events']:
        for moment in event['moments']:
            if total == 0:
                print(moment)

            quarter: str = str(moment[0])
            time: int = moment[2]

            found = False
            for offset in range(-3, 3):
                key = f"{quarter} {str(time + (offset / 100))}"
                if key in timestamps:
                    moment.append(timestamps[key])
                    found = True
                    break
            if not found:
                moment.append(NO_MATCH)  # no match
                misses += 1
            total += 1

            # Timestamp gets append to end of moment arr, not always going to be in a reliable pos.
            # Maybe stavu data should use more dicts to solve this problem

    print(f"found frames for {(total - misses)/ total}% of moments")
    with open(out_path, 'w') as f:
        json.dump(statvu_data, f)
