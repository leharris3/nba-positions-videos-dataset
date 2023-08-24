import json

MAX_TIME = 720.0
MIN_TIME = 0.0
MIN_QUARTER = 1
MAX_QUARTER = 4


def postprocess_timestamps(json_path: str) -> str:
    """Interpolate and remove noise from extracted timestamps."""

    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
    except:
        print(
            f"Error: could not open extracted timestamps at path {json_path}.")
        raise Exception

    modified_data = {}
    modified_times = {}
    frame_count = 0  # Maximum interpolation is one second

    for frame, time in data.items():
        quarter, remaining = time
        key = (quarter, remaining)

        # remove garbage results
        if remaining > MAX_TIME or remaining < MIN_TIME or int(quarter) > MAX_QUARTER or int(quarter) < MIN_QUARTER:
            pass
        else:
            if key in modified_times:
                if remaining != 0.0 and frame_count < 24:
                    modified_times[key] -= 0.04
                    modified_data[frame] = [
                        quarter, round(modified_times[key], 2)]
                    frame_count += 1
                else:
                    modified_data[frame] = time
            else:
                modified_times[key] = remaining
                modified_data[frame] = time
                frame_count = 0

    modified_json_path = json_path.rstrip('.json') + '_modified.json'

    with open(modified_json_path, 'w') as file:
        json.dump(modified_data, file, indent=4)

    print(f"Modified timestamp data has been saved to: {modified_json_path}.")
    return modified_json_path


# Usage example
# json_file_path = 'demo/full-game-example/results.json'
# modify_time_remaining(json_file_path)
