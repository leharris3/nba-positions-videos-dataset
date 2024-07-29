import logging
import torch

from typing import Dict

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ViTPoseCustom:
    """
    Custom model class for a easy_vit keypoint detection model.
    """

    def __init__(self, config: Dict, model: torch.nn.Module) -> None:
        self.config = config
        self.model = model

        # set to infer mode
        self.model.eval()

        # TODO: order matters
        if config["use_half_precision"] == "True":
            self.model = self.model.half()
        if config["compile_model"] == "True":
            self.model = torch.compile(self.model)

    def get_model(self, device: int) -> torch.nn.Module:
        """
        Call this method to return a ViTPose model object.
        We only create a copy of this model ONCE in shared memory.

        TODO: if we call this function from multiple processes, are we
        still creating a new copy of the model.
        """

        return self.model.to(device)
