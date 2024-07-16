import logging

from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Dict, Tuple


class Florence:

    _model = None
    _processor = None

    @staticmethod
    def get_model_and_processor(
        args: Dict,
    ) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
        logging.info(f"Loading model...")
        if Florence._model is not None:
            return Florence._model, Florence._processor
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base-ft",
            trust_remote_code=True,
        )
        logging.info(f"Loading processor...")
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base-ft",
            trust_remote_code=True,
        )
        Florence._model, Florence._processor = model, processor
        return model, processor
