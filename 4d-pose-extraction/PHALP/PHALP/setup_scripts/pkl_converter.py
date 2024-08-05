# Conversion starts from here   
import numpy as np
import os
import joblib 
import pickle
import sys

base_path = ''

def process_line(line, scale_factor=1.075):
    """Process a single line of the text file and return relevant data."""
    parts = line.strip().split(',')
    frame_id = int(parts[0]) + 1
    track_id = int(parts[1]) - 1  # Adjust if necessary
    x1, y1, width, height = map(float, parts[2:6])
    
    # Apply scaling factor to width and height
    scaled_width = width * scale_factor
    scaled_height = height * scale_factor
    
    # Adjust x1, y1 to maintain the center of the bbox
    x1 = x1 - (scaled_width - width) / 2
    y1 = y1 - (scaled_height - height) / 2
    
    # Create the bbox in the format x1, y1, width, height
    bbox = np.array([x1, y1, scaled_width, scaled_height], dtype=np.float32)
    
    return frame_id, track_id, bbox

def read_file(filename):
    """Read the file and return the data organized by frame, with gt_track_id handled correctly."""
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            frame_id, track_id, bbox = process_line(line)
            frame_name = os.path.join(base_path, f'{frame_id:06}.jpg')
            if frame_name not in data:
                data[frame_name] = {"gt_bbox": [], "extra_data": {"gt_class": [], "gt_track_id": []}}
            data[frame_name]["gt_bbox"].append(bbox.tolist())  # Convert bbox to list for JSON-like output
            if track_id not in data[frame_name]["extra_data"]["gt_track_id"]:
                data[frame_name]["extra_data"]["gt_track_id"].append(track_id)
    return data

def save_data(data, output_filename):
    """Save the processed data to a pickle file using joblib."""
    joblib.dump(data, output_filename)




input_filename = '/playpen-storage/masonm/pose-alignment/mixsort_exp/ex_2/mix_exp/2024_02_27_11_53_48.txt'

output_filename = '/playpen-storage/masonm/pose-alignment/mixsort_exp/ex_2/mix_exp/rockvjazz.pkl'

data = read_file(input_filename)
# convert_data_to_numpy(data)  # Call this only if you want numpy arrays instead of lists.
save_data(data, output_filename)

print(f'Data from {input_filename} has been processed and saved to {output_filename} using joblib.')
