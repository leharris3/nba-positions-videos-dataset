import json
import pandas as pd
from pathlib import Path
from post_processing.post_processing import get_unique_moments_from_statvu, map_frames_to_moments
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def define_paths():
    """Define and return the paths for the directories."""
    timestamps_dir = Path("/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/data/nba-15-16-timestamps-post-processed")
    statvu_dir = Path("/mnt/sun/levlevi/nba-plus-statvu-dataset/statvu-game-logs")
    out_dir = Path("/mnt/sun/levlevi/nba-plus-statvu-dataset/2d-player-positions")
    
    assert timestamps_dir.exists() and timestamps_dir.is_dir(), f"Directory not found: {timestamps_dir}"
    assert statvu_dir.exists() and statvu_dir.is_dir(), f"Directory not found: {statvu_dir}"
    assert out_dir.exists() and out_dir.is_dir(), f"Directory not found: {out_dir}"
    
    return timestamps_dir, statvu_dir, out_dir

def load_data(timestamps_dir, statvu_dir):
    """Load data from the given directories and return the file paths."""
    statvu_paths = list(statvu_dir.glob('*'))
    timestamps_paths = list(timestamps_dir.glob('*'))
    
    assert len(statvu_paths) > 0, f"No files found in directory: {statvu_dir}"
    assert len(timestamps_paths) > 0, f"No files found in directory: {timestamps_dir}"
    
    timestamp_ids = {fp.stem.split("_")[0]: fp for fp in timestamps_paths}
    return statvu_paths, timestamp_ids


def match_paths(statvu_paths, timestamp_ids):
    """
    Match statvu paths with timestamp paths based on file stem ids.
    
    Parameters:
        statvu_paths (list): List of statvu paths.
        timestamp_ids (dict): Dictionary of timestamp ids and their corresponding paths.
    
    Returns:
        pd.DataFrame: DataFrame containing matched timestamp and statvu paths.
    """

    statvu_matched_paths = []
    for sv_path in statvu_paths:
        if len(sv_path.suffix.split('.')) > 1:
            game_id = sv_path.suffix.split('.')[1]
            if game_id in timestamp_ids:
                statvu_matched_paths.append(sv_path)
    
    matched_data = {
        "timestamp_path": [timestamp_ids[sv_path.suffix.split('.')[1]] for sv_path in statvu_matched_paths],
        "statvu_dir_path": statvu_matched_paths
    }
    
    return pd.DataFrame(matched_data)


def process_file(row, out_dir):
    """
    Process a single file row and save the output to the specified directory.
    
    Parameters:
        row (pd.Series): A row containing paths for timestamp and statvu directories.
        out_dir (Path): The directory to save the processed file.
    """
    timestamp_path = Path(row['timestamp_path'])
    statvu_dir_path = Path(row['statvu_dir_path'])
    
    assert timestamp_path.exists(), f"Timestamp file not found: {timestamp_path}"
    assert statvu_dir_path.exists(), f"Statvu directory not found: {statvu_dir_path}"
    
    try:
        statvu_path = next(statvu_dir_path.glob("*.json"), None)
        if not statvu_path:
            logging.warning(f"No JSON file found in {statvu_dir_path}")
            return
        
        game_id = statvu_dir_path.stem
        quarter = timestamp_path.stem.split("_period")[-1][0]
        new_name = f"{game_id}.Q{quarter}.2D-POS.json"
        new_path = out_dir / new_name

        moments = get_unique_moments_from_statvu(statvu_path)
        
        with open(timestamp_path, "r") as f:
            timestamp_data = json.load(f)
        
        mapped_data = map_frames_to_moments(timestamp_data, moments)
        
        with open(new_path, "w") as f:
            json.dump(mapped_data, f, indent=4)
    except Exception as e:
        logging.error(f"Error: could not process video at {timestamp_path} due to {e}")


def main():
    timestamps_dir, statvu_dir, out_dir = define_paths()
    statvu_paths, timestamp_ids = load_data(timestamps_dir, statvu_dir)
    path_map = match_paths(statvu_paths, timestamp_ids)
    
    logging.info(f"Processing {len(path_map)} files.")
    logging.info(f"Estimated time: {20 * len(path_map)} seconds!")

    for _, row in tqdm(path_map.iterrows(), total=path_map.shape[0], desc="Processing files"):
        process_file(row, out_dir)

if __name__ == "__main__":
    main()
