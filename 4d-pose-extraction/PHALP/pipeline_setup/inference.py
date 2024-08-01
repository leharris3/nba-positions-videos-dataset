import os
import subprocess
import shutil
import cv2
from pathlib import Path



def run_phalp_inference(input_path, output_dir, gt_pkl_path=None):
    """
    Run PHALP inference on a video or a directory of frames.

    Args:
    input_path (str): Path to input video file or directory containing frame images.
    output_dir (str): Path to directory where output should be saved.
    gt_pkl_path (str, optional): Path to ground truth pickle file. If provided, will use ground truth.

    Returns:
    str: Path to the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine if input is video or frames
    if os.path.isfile(input_path):
        # Input is a video file
        video_path = input_path
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames from video
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(frames_dir, f"frame{count:06d}.jpg"), image)
            success, image = vidcap.read()
            count += 1
        vidcap.release()
        
        base_path = frames_dir
    else:
        # Input is a directory of frames
        base_path = input_path

    # Construct PHALP command
    cmd = [
        "/usr/bin/time", "-v",
        "python", "scripts/demo.py",
        "render.enable=True",
        f"video.output_dir={output_dir}",
        f"video.base_path={base_path}"
    ]

    if gt_pkl_path:
        cmd.extend([
            "use_gt=True",
            f"video.source={gt_pkl_path}"
        ])
    else:
        cmd.append("use_gt=False")

    # Run PHALP command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running PHALP: {e}")
        return None

    # Clean up temporary frames directory if we created one
    if os.path.isfile(input_path):
        shutil.rmtree(frames_dir)

    return output_dir
#example_runs:
"""
# For a video file with ground truth data:
output = run_phalp_inference("/path/to/video.mp4", "/path/to/output", "/path/to/gt.pkl")
# For a directory of frames with ground truth data:
output = run_phalp_inference("/path/to/frames_directory", "/path/to/output", "/path/to/gt.pkl")
# For a video file without ground truth data:
output = run_phalp_inference("/path/to/video.mp4", "/path/to/output")
# For a directory of frames without ground truth data:
output = run_phalp_inference("/path/to/frames_directory", "/path/to/output")
"""