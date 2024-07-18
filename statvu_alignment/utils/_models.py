import logging
import torch

from typing import Dict
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Dict, Tuple


class YOLOModel:

    _model = None

    @classmethod
    def get_model(cls, config: Dict):
        verbose = config["yolo"]["verbose"]
        if cls._model is None:
            cls._model = YOLO(config["yolo_model_path"], verbose=verbose)
        return cls._model


class Florence:

    _model = None
    _processor = None

    @staticmethod
    def get_model_and_processor(
        config: Dict,
    ) -> Tuple[AutoModelForCausalLM, AutoProcessor]:

        logging.info(f"Loading Florence-2 Model...")
        model_path = config["model_path"]
        processor_path = config["processor_path"]
        half = config["use_half_precision"]
        compile = config["compile_model"]

        if Florence._model is not None:
            return Florence._model, Florence._processor
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if half == "True":
            model = model.half()
        if compile == "True":
            model = torch.compile(model)

        logging.info(f"Loading processor...")
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        Florence._model, Florence._processor = model, processor
        return model, processor
