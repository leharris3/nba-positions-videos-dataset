export PYTHONPATH="${PYTHONPATH}:/mnt/arc/levlevi/nba-positions-videos-dataset/4d-pose-extraction/PHALP/PHALP"

# vid_src=/mnt/arc/levlevi/nba-positions-videos-dataset/4d-pose-extraction/PHALP/PHALP/assets/videos/gymnasts.mp4
vid_src=/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/filtered-clips/162142/period2/162142_period2_1+_76939328_0.mp4

# python scripts/demo_phalp.py \
# phalp.detector="maskrcnn" \
# verbose=False \
# debug=False \
# render.enable=True \
# video.output_dir=nba-demo \
# video.extract_video=True \
# use_gt=False \
# device=cuda \
# video.source=${vid_src} \

python scripts/parallel_proc_test.py --start_fp_idx 0 --end_fp_idx 2