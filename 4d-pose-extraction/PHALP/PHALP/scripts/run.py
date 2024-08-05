import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# os.chdir("PHALP/PHALP")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Base command
BASE_CMD = "python process_vids_parallel.py"
ANNOTATIONS_DIR = "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/filtered-clip-annotations"
NUM_GPUS = 1
PROC_PER_GPU = 2


def get_annotation_files(directory):
    """Retrieve all JSON annotation files from the given directory."""
    annotation_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(directory)
        for f in filenames
        if f.endswith(".json")
    ]
    return annotation_files


def calculate_indices(total_files, total_processes):
    """Calculate the start and end indices for each process."""
    files_per_process = total_files // total_processes
    remainder = total_files % total_processes
    indices = []

    for process_num in range(total_processes):
        start_idx = process_num * files_per_process
        end_idx = (process_num + 1) * files_per_process
        if process_num < remainder:
            start_idx += process_num
            end_idx += process_num + 1
        else:
            start_idx += remainder
            end_idx += remainder
        indices.append((start_idx, end_idx))

    return indices


def run_process(start_idx, end_idx, gpu):
    """Run the subprocess with the given indices and GPU."""
    cmd = f"{BASE_CMD} --start_fp_idx {start_idx} --end_fp_idx {end_idx}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
        logging.info(
            f"Process on GPU {gpu} with indices {start_idx} to {end_idx} completed successfully."
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Process on GPU {gpu} with indices {start_idx} to {end_idx} failed: {e}"
        )


def main(directory, num_gpus, processes_per_gpu):
    annotation_files = get_annotation_files(directory)
    total_files = len(annotation_files)
    total_processes = num_gpus * processes_per_gpu

    logging.info(f"Total files: {total_files}")
    logging.info(f"Total processes: {total_processes}")

    indices = calculate_indices(total_files, total_processes)

    process_count = 0

    # use a threadpool to launch these jobs
    with ThreadPoolExecutor(max_workers=total_processes) as executor:
        futures = []
        for gpu in range(num_gpus):
            for _ in range(processes_per_gpu):
                start_idx, end_idx = indices[process_count]
                futures.append(executor.submit(run_process, start_idx, end_idx, gpu))
                process_count += 1

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error occurred during processing: {e}")

    logging.info("All processes completed.")


if __name__ == "__main__":
    main(ANNOTATIONS_DIR, NUM_GPUS, PROC_PER_GPU)
