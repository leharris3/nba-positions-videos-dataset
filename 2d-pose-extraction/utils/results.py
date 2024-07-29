import json
import logging
import numpy as np
import os
import torch

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from utils.data import NBAClips
from utils.model import ViTPoseCustom
from easy_ViTPose.vit_utils.top_down_eval import keypoints_from_heatmaps


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_results(out_fp: str, results: Dict):
    with open(out_fp, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)
        
        
def process_grouped_result(config, fp, file_results):
    out_fp = os.path.join(config["results_dir"], *fp.split("/")[-3:])
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    curr_ann = NBAClips.load_annotations(fp)
    # update annotations
    for result, frame_idx, rel_bbx_idx in file_results:
        try:
            curr_ann["frames"][int(frame_idx)]["bbox"][rel_bbx_idx][
                "keypoints"
            ] = result
        except IndexError:
            # logger.error(f"Error writing results: {frame_idx}, {rel_bbx_idx}")
            pass  # silently ignore out-of-bounds errors

    # write results
    write_results(out_fp, curr_ann)


def group_result(annotation_fps, grouped_results, result, fp_idx, frame_idx, rel_bbx_idx):
    fp = annotation_fps[fp_idx]
    grouped_results[fp].append((result, frame_idx, rel_bbx_idx))


def update_results(
    config: Dict,
    results: List,
    annotation_fps: List[str],
    curr_annotation_fp_idx: List[int],
    curr_frame_idx: List[int],
    curr_rel_bbx_idx: List[int],
):
    # group results by annotation file path
    grouped_results = defaultdict(list)
    with ThreadPoolExecutor() as executor:
        for result, fp_idx, frame_idx, rel_bbx_idx in zip(
            results, curr_annotation_fp_idx, curr_frame_idx, curr_rel_bbx_idx
        ):
            executor.submit(
                group_result,
                annotation_fps,
                grouped_results,
                result,
                fp_idx,
                frame_idx,
                rel_bbx_idx,
            )
            
    # write all results to file paths
    with ThreadPoolExecutor() as executor:
        for fp, file_results in grouped_results.items():
            executor.submit(process_grouped_result, config, fp, file_results)
        return # no need to wait around for the results
       
