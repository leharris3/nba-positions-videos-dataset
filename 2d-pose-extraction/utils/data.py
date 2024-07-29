# <3
# stolen from: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/dataloaders/datasets/coco.py

import numpy as np
import torch
import os
import ujson as json
import sys
import time
import cv2
import logging

from glob import glob
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from tqdm import trange
from torchvision.io import read_video

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TARGET_SIZE = [192, 256]

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NBAClips(Dataset):
    def __init__(self, config: Dict, annotation_file_paths: List, device: str) -> None:
        super().__init__()
        self.config: Dict = config
        assert self.config is not None, f"Config file not found: {self.config}"
        
        self.annotations_dir: str = config["clips_annotations_dir"]
        assert self.annotations_dir is not None, f"Annotations dir not found: {self.annotations_dir}"
        assert os.path.isdir(self.annotations_dir), f"Annotations dir not found: {self.annotations_dir}"
        
        self.results_dir: str = config["results_dir"]
        assert self.results_dir is not None, f"Results dir not found: {self.results_dir}"
        assert os.path.isdir(self.results_dir), f"Results dir not found: {self.results_dir}"
        
        self.device: str = device

        # annotation file paths
        self.annotation_file_paths: List = annotation_file_paths
        self.remove_processed_annotations()

        # current video capture
        # TODO: use torch vision dataloader instead
        self.cap = None
        self.curr_frame = None

        # current annotation obj and bbx
        self.current_annotations: Optional[Dict] = None
        self.current_bbx: Optional[Dict] = None

        # abs idx of the current annotation file loaded
        self.current_annotation_fp_idx = 0
        self.current_frame_idx = 0

        # within an annotation file
        self.current_rel_bbx_idx = 0

    # TODO: ~.19s/16 elements
    def __getitem__(self, index):

        # no annoations file loaded -> new video from mem
        if self.current_annotations is None:
            # TODO: end of the dataset
            if self.current_annotation_fp_idx >= len(self.annotation_file_paths):
                return None
            
            # get next annotations fp
            current_annotation_fp = self.annotation_file_paths[
                self.current_annotation_fp_idx
            ]
            
            self.current_annotations = NBAClips.load_annotations(current_annotation_fp)
            # no more videos to process
            if len(self.current_annotations["frames"]) == 0:
                return None
            
            self.current_video_fp = self.get_video_fp(self.config["nba_ds_root_dir"])
            
            # error reading video -> load next video
            if self.current_video_fp is None:
                self.current_annotation_fp_idx += 1
                self.reset()
                return self.__getitem__(index)

            # load the next video
            try:
                self.cap = cv2.VideoCapture(self.current_video_fp)
            except:
                logger.error(f"Failed to load video: {self.current_video_fp}")
                self.reset()
                self.current_annotation_fp_idx += 1
                return self.__getitem__(index)
                
            ret, self.curr_frame = self.cap.read()
            if not ret:
                logger.error(f"Failed to read frame from: {self.current_video_fp}")
                self.reset()
                self.current_annotation_fp_idx += 1
                return self.__getitem__(index)

            # reset pointers
            self.current_frame_idx = 0
            self.current_rel_bbx_idx = 0

        # check for EOF
        total_frames = len(self.current_annotations["frames"])
        total_frames_cap = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # TODO: this bug would imply there is a fundamental error in the way we construct annotations
        try:
            assert total_frames == total_frames_cap
        except:
            logger.warning(f'annotations and video have different number of frames: {total_frames} != {total_frames_cap}')
        
        # frame idx is out of range of current annotations -> load new video
        if self.current_frame_idx >= total_frames:
            self.reset()
            self.current_annotation_fp_idx += 1
            return self.__getitem__(index)
        else:
            num_bbxs_in_current_frames = len(self.current_annotations["frames"][self.current_frame_idx]["bbox"])
            # current bbx is out of range -> load new frame
            if self.current_rel_bbx_idx >= num_bbxs_in_current_frames:
                self.current_rel_bbx_idx = 0
                self.current_frame_idx += 1
                ret, self.curr_frameframe = self.curr_frame = self.cap.read()
                # corrupt vid -> load new video
                if not ret:
                    self.reset()
                    self.current_annotation_fp_idx += 1
                    return self.__getitem__(index)
                
        # load the next bounding box
        try:
            self.current_bbx = self.current_annotations["frames"][
                self.current_frame_idx
            ]["bbox"][self.current_rel_bbx_idx]
        except IndexError:
            logger.warning(f"WARN: current bbx index out of range after checks: {self.current_rel_bbx_idx}")
            self.current_frame_idx += 1
            self.current_rel_bbx_idx = 0
            return self.__getitem__(index)

        try:
            x, y, w, h = map(
                int,
                (
                    self.current_bbx["x"],
                    self.current_bbx["y"],
                    self.current_bbx["width"],
                    self.current_bbx["height"],
                ),
            )
        except:
            logger.error(f"Failed to load bbx: {self.current_bbx}")
            self.current_rel_bbx_idx += 1
            return self.__getitem__(index)

        # crop the current frame
        try:
            curr_frame_cropped = self.curr_frame[y : y + h, x : x + w]
        except:
            logger.error(f"Failed to crop frame: {self.current_video_fp}")
            logger.error(f"bbx: {self.current_bbx}")
            self.current_rel_bbx_idx += 1
            return self.__getitem__(index)

        # resize + normalize
        try:
            curr_frame_pre_processed, og_h, og_w = NBAClips.pre(
                np.array(curr_frame_cropped)
            )
        except:
            logger.error(f"Failed to preprocess frame: {self.current_video_fp}")
            self.current_rel_bbx_idx += 1
            return self.__getitem__(index)

        # (1, W, H, C)
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

        # always increment the bbx pointer
        self.current_rel_bbx_idx += 1
        return curr_data_item

    def __len__(self):
        # bad hack
        return sys.maxsize
    
    def reset(self):
        self.current_frame_idx = 0
        self.current_rel_bbx_idx = 0
        self.current_annotations = None
        self.curr_frame = None
        self.cap = None
        self.current_bbx = None
        self.current_video_fp = None

    def load_frame(self, frame_idx):
        frame, _, _ = read_video(
            self.current_video_fp,
            start_pts=self.video_metadata[0][frame_idx],
            end_pts=self.video_metadata[0][frame_idx],
            pts_unit="sec",
        )
        return frame.squeeze(0).numpy()

    @staticmethod
    def load_annotations(fp: str) -> Optional[Dict]:
        # slightly faster way to read a json file
        try:
            with open(fp, "rb") as f:
                return json.load(f)
        except:
            logger.error(f"Failed to load annotations: {fp}")
            return None

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

        processed_annotations = set(
            [
                fp.replace(self.results_dir, self.annotations_dir)
                for fp in glob(os.path.join(self.results_dir, "*/*/*.json"))
            ]
        )
        self.annotation_file_paths = list(
            set(self.annotation_file_paths) - processed_annotations
        )
