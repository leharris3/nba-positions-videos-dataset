import random
import logging
import warnings
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import easyocr

# Configure warnings and logging
warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def add_noise(image: np.ndarray, std_dev: float = 3.0) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, std_dev, image.shape).astype(np.uint8)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def process_image(reader: easyocr.Reader, image_path: Path, num_bootstraps: int = 7) -> List:
    """Process a single image with bootstrapping."""
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        bootstraps = []
        for _ in range(num_bootstraps):
            noisy_image = add_noise(image_array)
            result = reader.readtext(noisy_image)
            bootstraps.append(result)
        return bootstraps
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return []

def ocr(config: Dict, image_file_paths: List[Path]) -> Dict[str, List]:
    """Perform OCR on a list of image files with bootstrapping."""
    reader = easyocr.Reader(["ch_sim", "en"])
    results = {}
    for image_file_path in tqdm(image_file_paths, desc="Processing images"):
        bootstraps = process_image(reader, image_file_path, config.get('num_bootstraps', 7))
        if bootstraps:
            results[str(image_file_path)] = bootstraps
        break
    return results