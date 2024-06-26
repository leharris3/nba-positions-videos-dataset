import json
import bisect
import numpy as np

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

def update_timestamps(timestamps, time_remaining):
    for k, v in enumerate(time_remaining):
        timestamps[str(k)]['time_remaining'] = v
    return timestamps


def get_timestamps_from_fp(fp):
    with open(fp, 'r') as f:
        timestamps = json.load(f)
    return timestamps


def get_time_remaining_from_timestamps_fp(fp):
    timestamps = get_timestamps_from_fp(fp)
    return get_time_remaining_from_timestamps(timestamps)


def get_time_remaining_from_timestamps(timestamps):
    return np.array([v['time_remaining'] if v['time_remaining'] is not None else 0 for v in timestamps.values()])


def post_process_timestamps(timestamps):

    timestamps = timestamps.copy()

    def extend_timestamps(time_remaining):
        """
        Interpolate timestamps in-place.
        """
        _time_remaining = []
        last_time = 0
        for val in time_remaining:
            if val != None and val > 1:
                last_time = val
            _time_remaining.append(last_time)
        return _time_remaining
    
    def interpolate(time_remaining):

        time_remaining = time_remaining.copy()
        fps = 30
        multiplier = 0
        decreasing = False
        for i in range(len(time_remaining) - 1):
            current, next_value = time_remaining[i], time_remaining[i + 1]
            peak_value = time_remaining[min(i + fps, len(time_remaining) - 1)]
            if current == 0:
                continue
            decreasing = peak_value < current
            if decreasing:
                if multiplier > 30:
                    multiplier, decreasing = 0, False
                    continue
                time_remaining[i] -= round((1/30) * multiplier, 2)
                multiplier = 0 if next_value < current else multiplier + 1
        return time_remaining

    def moving_average(x, window):
        return np.convolve(x, np.ones(window), 'valid') / window

    def normalize(arr):
        _min, _max = arr.min(), arr.max()
        return (arr - _min) / (_max - _min)

    def denoise_time_remaining(time_remaining):

        def update_time_remaining(remove_indices, time_remaining):
            valid_indices = np.where(remove_indices == 0)[0]
            for idx in np.where(remove_indices)[0]:
                nearest_valid_index = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                time_remaining[idx] = time_remaining[nearest_valid_index]

        # remove values that deviate too far from expected values
        time_remaining = np.array(time_remaining)
        time_remaining_og = time_remaining.copy()
        expected = np.linspace(100, 720, len(time_remaining), endpoint=False)[::-1]
        norm_expected_diff = normalize(np.abs(expected - time_remaining_og))
        remove_indices = (norm_expected_diff > 0.5).astype(int)
        update_time_remaining(remove_indices, time_remaining)

        # convolve with shrinking window
        for window in [1000, 500]:
            if len(time_remaining) > window:
                mvg_avg = moving_average(time_remaining, window)
                padded_avg = np.pad(mvg_avg, (window // 2, window - window // 2 - 1), mode='edge')
                norm_diff = normalize(np.abs(time_remaining - padded_avg))
                remove_indices = (norm_diff > 0.5).astype(int)
                update_time_remaining(remove_indices, time_remaining)

        # convolve with shrinking window
        for window in [50, 10, 5]:
            if len(time_remaining) > window:
                mvg_avg = moving_average(time_remaining, window)
                padded_avg = np.pad(mvg_avg, (window // 2, window - window // 2 - 1), mode='edge')
                norm_diff = normalize(np.abs(time_remaining - padded_avg))
                remove_indices = (norm_diff > 0.5).astype(int)
                update_time_remaining(remove_indices, time_remaining)

        temp_interpolated = interpolate(time_remaining)
        delta = np.gradient(temp_interpolated)
        delta_inter = normalize(moving_average(abs(delta), 7))
        remove_indices = (delta_inter > 0.1).astype(int)
        update_time_remaining(remove_indices, time_remaining)
        return time_remaining
    
    def remove_delta_zero(a, b):
        if len(a) != len(b):
            raise ValueError("The arrays 'a' and 'b' must be of equal length.")
        # Iterate through the arrays
        for i in range(len(a)):
            if b[i] == 0:
                a[i] = None
        return a

    time_remaining = get_time_remaining_from_timestamps(timestamps)
    extended_time_remaining = extend_timestamps(time_remaining)
    denoised_time_remaining = denoise_time_remaining(extended_time_remaining)
    interpolated_time_remaining = interpolate(denoised_time_remaining)

    # remove values where delta = 0
    delta_time_remaining = np.gradient(interpolated_time_remaining)
    remove_delta_zero(interpolated_time_remaining, delta_time_remaining)

    timestamps = update_timestamps(
        timestamps=timestamps,
        time_remaining=interpolated_time_remaining
    )
    return timestamps


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

    # Pre-process moments_data into a dictionary with lists of times for each quarter
    moments_dict = {}
    moments_values = {}
    for moment_key, moment_value in moments_data.items():
        quarter, time_remaining = map(float, moment_key.split('_'))
        quarter = str(int(quarter))
        if quarter not in moments_dict:
            moments_dict[quarter] = []
        moments_dict[quarter].append(time_remaining)
        moments_values[(quarter, time_remaining)] = moment_value

    # Sort the lists of times for each quarter to allow binary search
    for quarter in moments_dict:
        moments_dict[quarter].sort()

    frames_moments_map = {}
    frames_matched = 0
    total_frames = 0

    for frame_id, frame_data in data.items():
        quarter = str(frame_data['quarter'])
        time_remaining = frame_data.get('time_remaining')
        conf = frame_data.get('conf')
        moment = None

        if time_remaining is None:
            frames_moments_map[frame_id] = None
            continue
        if quarter not in moments_dict:
            raise ValueError(f"Invalid quarter: {quarter}")

        time_remaining = float(time_remaining)
        total_frames += 1

        moment_times = moments_dict[quarter]
        pos = bisect.bisect_left(moment_times, time_remaining)

        closest_time = None
        min_difference = float('inf')

        if pos < len(moment_times):
            if abs(time_remaining - moment_times[pos]) < min_difference:
                min_difference = abs(time_remaining - moment_times[pos])
                closest_time = moment_times[pos]
        if pos > 0:
            if abs(time_remaining - moment_times[pos - 1]) < min_difference:
                min_difference = abs(time_remaining - moment_times[pos - 1])
                closest_time = moment_times[pos - 1]

        if closest_time is not None:
            frames_matched += 1
            moment = moments_values[(quarter, closest_time)]

        entry = {
            'conf': conf,
            'pred_time_remaining': time_remaining,
            'pred_quarter': quarter,
            'moment': moment
        }
        frames_moments_map[frame_id] = entry

    return frames_moments_map