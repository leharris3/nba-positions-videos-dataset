import warnings
import json
import subprocess
import argparse
import os

from omegaconf import DictConfig
from phalp.utils import get_pylogger
from glob import glob
from phalp.trackers.PHALP import PHALP

warnings.filterwarnings("ignore")
log = get_pylogger(__name__)

DEFAULT_CONFIG = "/mnt/arc/levlevi/nba-positions-videos-dataset/4d-pose-extraction/PHALP/PHALP/scripts/config.json"
ANNOTATIONS_DIR = "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/filtered-clip-annotations"
PKL_OUT_DIR = "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/filtered-clip-phalp-outputs"

def get_video_fp(annotation_fp: str):
    video_fp = (
        annotation_fp.replace("filtered-clip-annotations", "filtered-clips")
        .replace("_annotation", "")
        .replace(".json", ".mp4")
    )
    if os.path.exists(video_fp):
        return video_fp
    else:
        return None


def main(args):

    # start and end idx
    start_idx = args.start_fp_idx
    end_idx = args.end_fp_idx

    # load config
    with open(DEFAULT_CONFIG, "r") as f:
        cfg = DictConfig(json.load(f))

    # load PHALP obj
    phalp_tracker = PHALP(cfg)

    # process a subset of annotations
    all_annotation_file_paths = sorted(glob(ANNOTATIONS_DIR + "/*/*/*.json"))[
        start_idx:end_idx
    ]
    
    # track all videos in the subset
    for fp in all_annotation_file_paths:
        video_fp = get_video_fp(fp)
        if video_fp is not None:
            try:
                pkl_out_fp = os.path.join(PKL_OUT_DIR, os.path.basename(video_fp).replace(".mp4", ".pkl"))
                PHALP.track(phalp_tracker, video_fp, pkl_out_fp)
            except Exception as e:
                print("Error processing video: ", video_fp)
                print(e)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--start_fp_idx", type=int, default=-1)
    args.add_argument("--end_fp_idx", type=int, default=-1)
    args = args.parse_args()
    main(args)
