import os
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf

def get_user_input():
    """Prompt the user for input paths."""
    video_path = input("Enter the path to the video file: ").strip()
    gt_tracks_path = input("Enter the path to the ground truth tracks pkl file: ").strip()
    output_dir = input("Enter the path for the output directory: ").strip()
    return video_path, gt_tracks_path, output_dir

def process_video_and_run_demo(video_path, output_dir, gt_tracks_path):
    """
    Process a video file by extracting frames and then run the PHALP demo script.

    Args:
    video_path (str): Path to the input video file
    output_dir (str): Path to the output directory for frames and results
    gt_tracks_path (str): Path to the ground truth tracks pkl file
    """
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create directories
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Extracting frames to {frames_dir}...")
    # Extract frames using ffmpeg
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-q:v", "2",
        os.path.join(frames_dir, "%06d.jpg")
    ]
    subprocess.run(ffmpeg_command, check=True)
    print("Frame extraction complete.")

    # Prepare configuration for the demo script
    @hydra.main(version_base="1.2", config_name="config")
    def run_demo(cfg: DictConfig):
        cfg.render.enable = True
        cfg.video.output_dir = os.path.join(output_dir, "test_gt_bbox")
        cfg.use_gt = True
        cfg.video.base_path = frames_dir
        cfg.video.source = gt_tracks_path

        print("Running PHALP demo script...")
        # Import and run the demo script
        from scripts.demo import main as demo_main
        demo_main(cfg)

    # Run the demo script
    run_demo()
    print("PHALP demo script execution complete.")

if __name__ == "__main__":
    video_path, gt_tracks_path, output_dir = get_user_input()
    
    print("\nProcessing with the following parameters:")
    print(f"Video path: {video_path}")
    print(f"Ground truth tracks path: {gt_tracks_path}")
    print(f"Output directory: {output_dir}")
    print()

    process_video_and_run_demo(video_path, output_dir, gt_tracks_path)
    print("\nAll processing complete.")