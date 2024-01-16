from moviepy.editor import VideoFileClip, clips_array

vid_path = '/Users/leviharris/Library/CloudStorage/GoogleDrive-leviharris555@gmail.com/Other computers/mac_new/NBA_HUDL_data/nba-plus-statvu-dataset/game-replays/720/18021_11-18-2015_2_Golden State Warriors_78_Toronto Raptors_period1.mp4'

def concatenate_videos(video_path1, video_path2, output_path):
    # Load the videos
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    # Resize videos to match their heights
    height = min(clip1.size[1], clip2.size[1])
    clip1_resized = clip1.resize(height=height)
    clip2_resized = clip2.resize(height=height)

    # Concatenate videos side-by-side
    final_clip = clips_array([[clip1_resized, clip2_resized]])

    # Write the result to a file
    final_clip.write_videofile(output_path, codec='libx264')

# Example usage
concatenate_videos(vid_path, 'test.mp4', 'concat_viz.mp4')