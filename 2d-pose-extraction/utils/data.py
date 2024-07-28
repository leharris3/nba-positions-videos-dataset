# <3
# stolen from: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/dataloaders/datasets/coco.py

import numpy as np
import torch
import os
import ujson as json
import sys
import time
import cv2


from glob import glob
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from tqdm import trange
from torchvision.io import read_video

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TARGET_SIZE = [192, 256]


class NBAClips(Dataset):
    def __init__(self, config: Dict, annotation_file_paths, device: str) -> None:
        super().__init__()
        self.config = config
        self.annotations_dir = config["clips_annotations_dir"]
        self.results_dir = config["results_dir"]
        self.device = device

        # annotation file paths
        self.annotation_file_paths = annotation_file_paths
        self.remove_processed_annotations()

        # map of {abs_bbx_idx: BoundingBox}
        # how big is the obj. going to be?
        self.bbx_map = {}

        # current video capture
        # TODO: use torch vision dataloader instead
        self.cap = None
        self.curr_frame = None

        # current annotation obj and bbx
        self.current_annotations = None
        self.current_bbx = None

        # abs idx of the current annotation file loaded
        self.current_annotation_fp_idx = 0
        self.current_frame_idx = 0

        # within an annotation file
        self.current_rel_bbx_idx = 0

        # global bounding box pointer
        self.current_abs_bbx_idx = 0

    # TODO: ~.19s/16 elements
    def __getitem__(self, index):

        # TODO: load video from mem + annotations obj from mem, del everything else
        if self.current_annotations is None:
            # congrats, we reached the end of the dataset
            if self.current_annotation_fp_idx >= len(self.annotation_file_paths):
                return None

            # get next annotations fp
            current_annotation_fp = self.annotation_file_paths[
                self.current_annotation_fp_idx
            ]
            self.current_annotations = NBAClips.load_annotations(current_annotation_fp)
            self.current_video_fp = self.get_video_fp(self.config["nba_ds_root_dir"])

            # try again
            if self.current_video_fp is None:
                self.current_annotation_fp_idx += 1
                self.current_annotations = None
                return self.__getitem__(index)

            # load the next video
            self.cap = cv2.VideoCapture(self.current_video_fp)
            ret, self.curr_frame = self.cap.read()

            # reset pointers
            self.current_frame_idx = 0
            self.current_rel_bbx_idx = 0

            # check for EOF

        # check for EOF
        total_frames = len(self.current_annotations["frames"])
        num_bbxs_in_current_frames = len(
            self.current_annotations["frames"][self.current_frame_idx]["bbox"]
        )
        #
        if self.current_rel_bbx_idx >= num_bbxs_in_current_frames:
            self.current_rel_bbx_idx = 0
            self.current_frame_idx += 1
            self.curr_frame = self.cap.read()[1]
            if self.current_frame_idx >= total_frames or self.curr_frame is None:
                self.current_frame_idx = 0
                self.current_annotation_fp_idx += 1
                self.current_annotations = None
                return self.__getitem__(index)

        # load the next bounding box
        try:
            self.current_bbx = self.current_annotations["frames"][
                self.current_frame_idx
            ]["bbox"][self.current_rel_bbx_idx]
        except IndexError:
            self.current_frame_idx += 1
            self.current_rel_bbx_idx = 0
            return self.__getitem__(index)

        x, y, w, h = map(
            int,
            (
                self.current_bbx["x"],
                self.current_bbx["y"],
                self.current_bbx["width"],
                self.current_bbx["height"],
            ),
        )

        # crop the current frame
        curr_frame_cropped = self.curr_frame[y : y + h, x : x + w]

        # resize + normalize
        curr_frame_pre_processed, og_h, og_w = NBAClips.pre(
            np.array(curr_frame_cropped)
        )

        # copy tensor -> GPU
        curr_frame_pre_processed = torch.tensor(curr_frame_pre_processed).squeeze()

        # cast to fp16
        if self.config["use_half_precision"] == "True":
            curr_frame_pre_processed = curr_frame_pre_processed.half()

        # curr obj to be returned
        curr_data_item = (
            curr_frame_pre_processed,
            self.current_annotation_fp_idx,
            self.current_frame_idx,
            self.current_rel_bbx_idx,
            og_h,
            og_w,
        )

        # update bbx pointers
        # always need to do this before checking for EOF
        self.current_abs_bbx_idx += 1
        self.current_rel_bbx_idx += 1

        # print(f"time to get item: {time.time() - start}")
        return curr_data_item

    def __len__(self):
        # bad hack
        return sys.maxsize

    def load_frame(self, frame_idx):
        frame, _, _ = read_video(
            self.current_video_fp,
            start_pts=self.video_metadata[0][frame_idx],
            end_pts=self.video_metadata[0][frame_idx],
            pts_unit="sec",
        )
        return frame.squeeze(0).numpy()

    @staticmethod
    def load_annotations(fp: str) -> Dict:
        # slightly faster way to read a json file
        with open(fp, "rb") as f:
            return json.load(f)

    @staticmethod
    def pre(img):
        try:
            org_h, org_w = img.shape[:2]
            img_input = (
                cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR) / 255
            )
            img_input = (
                ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
            )
        except:
            img = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3))
            org_h, org_w = img.shape[:2]
            img_input = (
                cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR) / 255
            )
            img_input = (
                ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
            )
        return img_input, org_h, org_w

    def get_video_fp(self, root_dir: str) -> Optional[str]:
        video_path = root_dir + self.current_annotations["video_path"]
        if not os.path.exists(video_path):
            return None
        return video_path

    def remove_processed_annotations(self) -> List[str]:
        """
        Return a list of all remaining annotation file paths to be processed.
        """
        to_process_names = {
            "_".join(os.path.basename(x).split("_")[0:-2]): x
            for x in self.annotation_file_paths
        }
        already_processed_names = [
            "_".join(os.path.basename(x).split("_")[0:-2])
            for x in glob(self.results_dir + "/*/*/*.json")
        ]
        for name in already_processed_names:
            if name in to_process_names:
                del to_process_names[name]
        self.annotation_file_paths = list(to_process_names.values())
