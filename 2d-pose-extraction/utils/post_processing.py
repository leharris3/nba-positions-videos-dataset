import logging
import numpy as np
import torch

from typing import Dict, List, Optional
from easy_ViTPose.vit_utils.top_down_eval import keypoints_from_heatmaps


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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