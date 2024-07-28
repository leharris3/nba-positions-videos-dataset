import json
import logging
import numpy as np
import os

from typing import Dict, List, Optional
from utils.data import NBAClips
from utils.model import ViTPoseCustom
from easy_ViTPose.vit_utils.top_down_eval import keypoints_from_heatmaps


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def postprocess(heatmaps, org_w, org_h):
    """
    Postprocess the heatmaps to obtain keypoints and their probabilities.

    Args:
        heatmaps (ndarray): Heatmap predictions from the model.
        org_w (int): Original width of the image.
        org_h (int): Original height of the image.

    Returns:
        ndarray: Processed keypoints with probabilities.
    """
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[org_w // 2, org_h // 2]]),
        scale=np.array([[org_w, org_h]]),
        unbiased=True,
        use_udp=True,
    )
    return np.concatenate([points[:, :, ::-1], prob], axis=2)


def process_hm(args):
    hm, w, h = args
    try:
        return postprocess(hm[np.newaxis], w, h)
    except:
        logger.error(f"Error processing: {args}")
        return None


def write_results(out_fp: str, results: Dict):
    with open(out_fp, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)


def update_results(
    config: Dict,
    results,
    annotation_fps: List[str],
    curr_annotation_fp_idx,
    curr_frame_idx,
    curr_rel_bbx_idx,
):
    curr_ann = None
    curr_fp = None
    for result, fp_idx, frame_idx, rel_bbx_idx in zip(
        results, curr_annotation_fp_idx, curr_frame_idx, curr_rel_bbx_idx
    ):
        fp = annotation_fps[fp_idx]
        out_fp = config["results_dir"] + "/" + "/".join(fp.split("/")[-3:])
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        if fp != curr_fp:
            if curr_fp is not None:
                write_results(out_fp, curr_ann)
            curr_ann = NBAClips.load_annotations(fp)
            curr_fp = fp
        # TODO: running into OOB errors
        try:
            curr_ann["frames"][int(frame_idx)]["bbox"][rel_bbx_idx][
                "keypoints"
            ] = result
        except:
            pass
    # write results to out
    if curr_ann is not None and curr_fp is not None:
        out_fp = config["results_dir"] + "/" + "/".join(fp.split("/")[-3:])
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        write_results(out_fp, curr_ann)
