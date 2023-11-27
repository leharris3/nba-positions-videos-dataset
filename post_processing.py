import json
import tqdm
import math

def get_unique_moments_from_statvu(statvu_log_path):
    """
    Extracts unique moments from a StatVu log file.

    Args:
    statvu_log_path (str): The file path of the StatVu log.

    Returns:
    dict: A dictionary mapping unique quarter-time remaining combinations to their respective moment details.
    """
    with open(statvu_log_path, 'r') as file:
        data = json.load(file)

    unique_quarter_time_combinations = set()
    processed_moments = {}

    for event in data['events']:
        for moment in event['moments']:
            quarter, moment_id, time_remaining_quarter, time_remaining_shot_clock, _, positions = moment
            moment_identifier = f"{quarter}_{time_remaining_quarter}"
            if moment_identifier not in unique_quarter_time_combinations:
                player_positions = [
                    {'team_id': player_data[0], 'player_id': player_data[1], 
                     'x_position': player_data[2], 'y_position': player_data[3], 'z_position': player_data[4]}
                    for player_data in positions
                ]
                processed_moments[moment_identifier] = {
                    'quarter': quarter,
                    'moment_id': moment_id,
                    'time_remaining_in_quarter': time_remaining_quarter,
                    'time_remaining_on_shot_clock': time_remaining_shot_clock,
                    'player_positions': player_positions
                }
                unique_quarter_time_combinations.add(moment_identifier)

    return processed_moments

def interpolate_timestamps(file_path, output_path):
    """
    Interpolates time remaining values in a JSON file containing timestamp data.

    Args:
    file_path (str): The file path of the input JSON file with timestamp data.
    output_path (str): The file path for the output JSON file with interpolated timestamps.

    The function reads the timestamp data, interpolates the 'time_remaining' values, and writes
    the updated data to a new JSON file.
    """

    # Load data from the file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Initialize list to store original time_remaining values
    original_time_values = []

    # Extract time_remaining values and handle None values
    for key in data:
        time_remaining = data[key].get('time_remaining', -1)
        original_time_values.append(time_remaining if time_remaining is not None else -1)

    # Initialize variables for interpolation
    last_seen_valid_time = -1
    consecutive_frame_count = 0
    interpolated_values = []

    # Perform interpolation of time_remaining values
    for value in original_time_values:
        if value > 0:
            if consecutive_frame_count == 0 or consecutive_frame_count > 30:
                last_seen_valid_time = value
                consecutive_frame_count = 0
            else:
                multiplier = math.floor((consecutive_frame_count / 30) * 25)
                interpolated_value = round(value - (multiplier / 25), 2)
                interpolated_values.append(interpolated_value)
            consecutive_frame_count += 1
        else:
            interpolated_values.append(None)

    # Update the data dictionary with interpolated values
    for key, interpolated_value in zip(data, interpolated_values):
        data[key]['time_remaining'] = interpolated_value

    # Write the updated data to a new JSON file
    try:
        with open(output_path, "w") as outfile:
            json.dump(data, outfile, indent=4)
    except Exception as e:
        print(f"Error writing file {output_path}: {e}")

def map_frames_to_moments(data, moments_data):
    """
    Maps frames in 'data' to their corresponding moments in 'moments_data' based on time proximity.

    Args:
    data (dict): A dictionary containing data with keys indicating frame identifiers and values 
                 having 'quarter' and 'time_remaining' information.
    moments_data (dict): A dictionary where keys are string representations of 'quarter_time' and values
                         are the moments to map.

    Returns:
    dict: A dictionary mapping frame identifiers to corresponding moments in 'moments_data'.
    """

    def is_close(time1, time2, tolerance=0.2 ):
        """Check if two time values are within a given tolerance."""
        return abs(time1 - time2) <= tolerance

    frames_matched = 0
    total_frames = 0
    frames_moments_map = {}

    moments_dict = {}
    for moment_key in moments_data:
        quarter, time_remaining = map(float, moment_key.split('_'))
        if quarter not in moments_dict:
            moments_dict[quarter] = []
        moments_dict[quarter].append(time_remaining)

    for frame_id in tqdm.tqdm(data):
        quarter_time_key = str(data[frame_id]['quarter']) + '_' + str(data[frame_id]['time_remaining']) if data[frame_id]['time_remaining'] != None else None
        if quarter_time_key:
            total_frames += 1
            quarter, time_remaining = map(float, quarter_time_key.split('_'))

            match_found = False
            if quarter in moments_dict:
                closest_time = None
                min_difference = float('inf')
                for moment_time in moments_dict[quarter]:
                    difference = abs(time_remaining - moment_time)
                    if difference < min_difference:
                        min_difference = difference
                        closest_time = moment_time

                if is_close(time_remaining, closest_time):
                    frames_matched += 1
                    match_found = True
                    moment_key = f"{int(quarter)}_{closest_time}"
                    frames_moments_map[frame_id] = moments_data[moment_key]
            if not match_found:
                frames_moments_map[frame_id] = None
        else:
            frames_moments_map[frame_id] = None

    print(frames_matched, '/', total_frames)
    return frames_moments_map
