import sys
import json
from typing import Dict, Any, List, Union
from data import Data
from timestamps import Timestamps
from utilities.files import File

NO_MATCH = -1


def print_progress(progress: float) -> None:
    """Print a progress bar."""

    progress_bar = "[" + "#" * \
        int(progress * 20) + " " * (20 - int(progress * 20)) + "]"
    sys.stdout.write("\r{} {:.2f}%".format(progress_bar, progress * 100))
    sys.stdout.flush()


def map_timestamps_to_statvu(timestamps: Timestamps, data: Data) -> str:
    """Maps a frame parameter to every moment in orginal stavu data file."""

    raw_data = data.get_data()
    total, misses = 0, 0
    quarter_time_timestamps_map = timestamps.get_timestamps_quarter_time_map()

    for event in raw_data["events"]:
        for moment in event["moments"]:
            quarter, time_remaining = str(moment[0]), moment[2]
            found = False
            for offset in range(-3, 3):
                key = f"{quarter} {str(time_remaining + (offset / 100))}"
                if key in quarter_time_timestamps_map:
                    moment.append(int(quarter_time_timestamps_map[key]))
                    found = True
                    print(moment)
                    break
            if not found:
                moment.append(NO_MATCH)
                misses += 1
            total += 1

    out_path = f"{data.path.strip('.json')}_with_timestamps.json"
    File.save_json(raw_data, out_path)

    print(f"found frames for {(total - misses)/ total}% of moments")
    return out_path
