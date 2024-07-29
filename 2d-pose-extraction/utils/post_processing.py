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


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def postprocess(heatmaps, org_w, org_h) -> np.ndarray:
    """
    Postprocess the heatmaps to obtain keypoints and their probabilities.

    Args:
        heatmaps (ndarray): Heatmap predictions from the model.
        org_w (int): Original width of the image.
        org_h (int): Original height of the image.

    Returns:
        ndarray: Processed keypoints with probabilities.
    """

    # TODO: parallel post-processing
    # TODO: not doing post-processing ATM
    # post-processing smooths results for the same person, but also dramtically increases processing time
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[org_w // 2, org_h // 2]]),
        scale=np.array([[org_w, org_h]]),
        unbiased=True,
        use_udp=True,
    )
    return np.concatenate([points[:, :, ::-1], prob], axis=2)


def process_hm(args) -> Optional[np.ndarray]:
    """
    Process a single heatmap.
    """

    hm, w, h = args
    try:
        return postprocess(hm[np.newaxis], w, h)
    except Exception as e:
        logger.error(f"Error processing: {hm.shape}, {w}, {h}")
        logger.error(e)
        return None


def post_process_results(
    heatmaps: torch.Tensor, og_w: torch.Tensor, og_h: torch.Tensor, device
) -> List[np.ndarray]:
    """
    Postprocess the heatmaps to obtain keypoints and their probabilities.

    Args:
        heatmaps (List[ndarray]): Heatmap predictions from the model.
        og_w (int): Original width of the image.
        og_h (int): Original height of the image.

    Returns:
        List[ndarray]: Processed keypoints with probabilities.
    """

    # heatmaps is a torch tensor copied to `device`
    # og_w and og_h are torch tensors on the CPU

    # TODO: default batch post-processing is ungodly slow!!!
    logger.debug(f"Postprocessing {heatmaps.shape} heatmaps")
    logger.debug(f"og_w: {og_w.shape}, og_h: {og_h.shape}")
    logger.debug(f"og_w: {og_w}, og_h: {og_h}")

    centers_arr = np.array([[x1, y1] for x1, y1 in zip(og_w // 2, og_h // 2)])
    scales_arr = np.array([[x1, y1] for x1, y1 in zip(og_w, og_h)])

    # TODO: unbiased and use_udp must be set to `True`
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=centers_arr,
        scale=scales_arr,
        unbiased=True,
        use_udp=True,
        device=device,
    )
    # probs need to be copied -> CPU
    points, prob = points, prob.cpu().numpy()
    return list(np.concatenate([points[:, :, ::-1], prob], axis=2))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_results(out_fp: str, results: Dict):
    with open(out_fp, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)


def update_results(
    config: Dict,
    results: List,
    annotation_fps: List[str],
    curr_annotation_fp_idx: List[int],
    curr_frame_idx: List[int],
    curr_rel_bbx_idx: List[int],
):
    # group results by file path
    grouped_results = defaultdict(list)
    for result, fp_idx, frame_idx, rel_bbx_idx in zip(
        results, curr_annotation_fp_idx, curr_frame_idx, curr_rel_bbx_idx
    ):
        fp = annotation_fps[fp_idx]
        grouped_results[fp].append((result, frame_idx, rel_bbx_idx))
    # process each file
    for fp, file_results in grouped_results.items():
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
