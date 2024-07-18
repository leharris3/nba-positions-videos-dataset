import torch
import logging
import warnings

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple
from PIL import Image
from typing import Dict

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ocr(
    rank,
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

    bootstraped_results = []
    logger.debug(f"Using device: {rank}")
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
    batch_size = config["ocr"]["batch_size"]
    total_batches = (len(images) + batch_size - 1) // batch_size

    # process data in batches
    for batch_idx in tqdm(range(total_batches), desc="Performing OCR on images"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(images))
        batch_images = images[start_idx:end_idx]
        logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")

        # pre-process image and text inputs
        prompts = [config["ocr"]["prompt"]] * len(batch_images)
        inputs = processor(text=prompts, images=batch_images, return_tensors="pt")
        input_ids = inputs["input_ids"].to(rank, non_blocking=True)
        pixel_values = inputs["pixel_values"].to(rank, non_blocking=True)

        # optionaly use half precision
        if config["use_half_precision"] == "True":
            pixel_values = pixel_values.half()
        del inputs
        logger.debug("Inputs moved to device and processed")

        # forward pass
        with torch.no_grad():
            logger.debug("Starting model generation")
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=config["ocr"]["new_max_tokens"],
                do_sample=config["ocr"]["do_sample"],
                early_stopping=config["ocr"]["early_stopping"],
                num_beams=config["ocr"]["num_bootstraps"],
                num_return_sequences=config["ocr"]["num_bootstraps"],
            )
        logger.debug("Model generation completed")

        # decode model outputs
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        logger.debug("Generated text decoded")

        # post-process outputs and append results to `bootstraped_results`
        for label, image in zip(generated_text, batch_images):
            parsed_answer = processor.post_process_generation(
                label, task="<OCR>", image_size=(image.width, image.height)
            )
            bootstraped_results.append(parsed_answer)
            logger.debug(f"Post-processed generation for image {image.filename}")

    return bootstraped_results if bootstraped_results else None
