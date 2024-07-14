import torch
import logging
import warnings

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FlorenceModel:

    # ensure we only load our model into memory once
    _model = None
    _processor = None
    _compile_model = None
    _model_variant = None
    _half = None

    @staticmethod
    def load_model_and_tokenizer(
        config: Optional[Dict],
    ) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
        if config is None:
            raise ValueError("Config cannot be None")
        if (
            FlorenceModel._model is not None
            and FlorenceModel._processor is not None
            and FlorenceModel._compile_model == config['florence']["compile_model"]
            and FlorenceModel._model_variant == config['florence']["model_variant"]
            and FlorenceModel._half == config['florence']["half"]
        ):
            return FlorenceModel._model, FlorenceModel._processor
        compile_model = config['florence']["compile_model"]
        model_variant = config['florence']["model_variant"]
        half = config['florence']["half"]
        try:
            logger.info("Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(
                f"microsoft/Florence-2-{model_variant}-ft",  # either 'base' or 'large'
                trust_remote_code=True,
                device_map="cuda",
            ).eval()
            processor = AutoProcessor.from_pretrained(
                f"microsoft/Florence-2-{model_variant}-ft",
                trust_remote_code=True,
            )
            if compile_model == "True":
                model = torch.compile(model)
            if half:
                model = model.half()
            FlorenceModel._model = model
            FlorenceModel._processor = processor
            FlorenceModel._compile_model = compile_model
            FlorenceModel._model_variant = model_variant
            FlorenceModel._half = half
            return model, processor
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise e


def ocr(
    config,
    image_file_paths: List[str],
    model,
    processor,
) -> Optional[List[str]]:
    def load_image(fp):
        logger.debug(f"Loading image {fp}")
        try:
            image = Image.open(fp)
            image.load()
            logger.debug(f"Successfully loaded image {fp}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {fp}: {e}")
            return None

    device = config["device"]
    logger.debug(f"Using device: {device}")
    bootstraped_results = []
    logger.debug("Starting to load images")
    # load all images using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_file_paths))
    images = [img for img in images if img is not None]
    if not images:
        logger.error("No valid images loaded.")
        return None
    logger.debug("Images loaded successfully")
    # define batch size
    batch_size = config['ocr']["batch_size"]
    total_batches = (len(images) + batch_size - 1) // batch_size
    # process images in batches
    for batch_idx in tqdm(range(total_batches), desc="Performing OCR on images"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(images))
        batch_images = images[start_idx:end_idx]
        prompts = [config['ocr']["prompt"]] * len(batch_images)
        logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")
        inputs = processor(text=prompts, images=batch_images, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device, non_blocking=True)
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True).half()
        del inputs
        logger.debug("Inputs moved to device and processed")
        with torch.no_grad():
            logger.debug("Starting model generation")
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=config['ocr']["new_max_tokens"],
                do_sample=config['ocr']["do_sample"],
                early_stopping=config['ocr']["early_stopping"],
                num_beams=config['ocr']["num_bootstraps"],
                num_return_sequences=config['ocr']["num_bootstraps"],
            )
        logger.debug("Model generation completed")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        logger.debug("Generated text decoded")
        for gt, image in zip(generated_text, batch_images):
            parsed_answer = processor.post_process_generation(
                gt, task="<OCR>", image_size=(image.width, image.height)
            )
            bootstraped_results.append(parsed_answer)
            logger.debug(f"Post-processed generation for image {image.filename}")
    logger.debug("OCR process completed")
    return bootstraped_results if bootstraped_results else None
