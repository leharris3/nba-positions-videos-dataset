import sys
import json
from typing import Dict, Any, List, Union
from data import Data
from timestamps import Timestamps
from files import File

NO_MATCH = -1


def print_progress(progress: float) -> None:
    """Print a progress bar."""

    progress_bar = "[" + "#" * \
        int(progress * 20) + " " * (20 - int(progress * 20)) + "]"
    sys.stdout.write("\r{} {:.2f}%".format(progress_bar, progress * 100))
    sys.stdout.flush()


def map_timestamps_to_statvu(timestamps: Timestamps, data: Data) -> str:
    """Maps a frame parameter to every moment in orginal stavu data file."""

    mapped_data = {}
    raw_data = data.get_data()
    timestamps_path = timestamps.get_path()
    try:
        timestamp_data_raw = json.load(open(timestamps_path))
    except:
        raise Exception(
            f"Error: could not load in timestamps from path: {timestamps_path}.")

    total, misses = 0, 0
    quarter_time_timestamps_map = timestamps.get_timestamps_quarter_time_map()
    for event in raw_data["events"]:
        for moment in event["moments"]:
            quarter, time_remaining = str(moment[0]), moment[2]
            found = any(
                (key := f"{quarter} {str(time_remaining + (offset * .01))}") in quarter_time_timestamps_map
                and moment.append(mapped_data[key])
                or False
                for offset in range(-3, 3)
            )
            if not found:
                moment.append(NO_MATCH)
                misses += 1
            total += 1

    out_path = f"{data.path.strip('.json')}_with_timestamps.json"
    File.save_json(mapped_data, out_path)

    print(f"found frames for {(total - misses)/ total}% of moments")
    return out_path
