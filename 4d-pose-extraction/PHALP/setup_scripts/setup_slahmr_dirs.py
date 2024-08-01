"""
python setup_slahmr_dirs.py /path/to/pkl/files /path/to/mp4/files /path/to/output/directory
"""
import os
import shutil
import argparse

def setup_directories(pkl_dir, video_dir, output_dir):
    # Ensure output directories exist
    phalp_out_dir = os.path.join(output_dir, 'slahmr', 'phalp_out')
    videos_dir = os.path.join(output_dir, 'slahmr', 'videos')
    os.makedirs(phalp_out_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Get list of pkl files
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

    for pkl_file in pkl_files:
        # Extract name without extension
        name = os.path.splitext(pkl_file)[0]
        
        # Create directory in phalp_out
        os.makedirs(os.path.join(phalp_out_dir, name), exist_ok=True)
        
        # Move pkl file
        shutil.copy2(os.path.join(pkl_dir, pkl_file), 
                     os.path.join(phalp_out_dir, name, pkl_file))
        
        # Look for corresponding mp4 file
        mp4_file = f"{name}.mp4"
        if os.path.exists(os.path.join(video_dir, mp4_file)):
            # Move mp4 file
            shutil.copy2(os.path.join(video_dir, mp4_file), 
                         os.path.join(videos_dir, mp4_file))
        else:
            print(f"Warning: No corresponding mp4 file found for {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup directories for SLAHMR project")
    parser.add_argument("pkl_dir", help="Directory containing .pkl files")
    parser.add_argument("video_dir", help="Directory containing .mp4 files")
    parser.add_argument("output_dir", help="Output directory for the organized structure")
    args = parser.parse_args()

    setup_directories(args.pkl_dir, args.video_dir, args.output_dir)
    print("Directory setup complete!")