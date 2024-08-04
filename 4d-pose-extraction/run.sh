cd /mnt/arc/levlevi/nba-positions-videos-dataset/4d-pose-extraction/VIBE
video_fp=/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/clips/707/period1/707_period1_1-_76423346.mp4
python demo.py --vid_file $video_fp --output_folder output/ --vibe_batch_size 64 --tracker_batch_size 64 --no_render