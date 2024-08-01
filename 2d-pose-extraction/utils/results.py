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
    

def compress_json_file(input_file: str, scale_factor: int = 100):
    
    with open(input_file, 'rb') as f:
        try:
            data = json.loads(f.read())
        except:
            print(f"error reading: {input_file}")
            return
    
    for frame in data['frames']:
        for bbx in frame['bbox']:
            bbx['x'] = int(bbx['x'])
            bbx['y'] = int(bbx['y'])
            bbx['width'] = int(bbx['width'])
            bbx['height'] = int(bbx['height'])
            bbx['confidence'] = int(bbx['confidence'] * scale_factor)
            if 'keypoints' in bbx:
                if len(bbx['keypoints']) > 1:
                    for kp in bbx['keypoints']:
                        try:
                            kp[0] = int(kp[0])
                            kp[1] = int(kp[1])
                            kp[2] = int(kp[2] * scale_factor)
                        except Exception as e:
                            print(f"error: {kp}")
                            print(e)
                            pass
                else:
                    for kp in bbx['keypoints'][0]:
                        try:
                            kp[0] = int(kp[0])
                            kp[1] = int(kp[1])
                            kp[2] = int(kp[2] * scale_factor)
                        except Exception as e:
                            print(f"error: {kp}")
                            print(e)
                            
    with open(input_file, 'wb') as f:
        f.write(json.dumps(data))


def write_results(out_fp: str, results: Dict):
    with open(out_fp, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)
        compress_json_file(out_fp)
        
        
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