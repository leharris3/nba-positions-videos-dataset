import cv2
import random
import logging
import warnings
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

# Configure warnings and logging
warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def resize_image(
    image: np.ndarray, target_height: int = 100, augment_factor: float = 0.1
) -> np.ndarray:
    """
    Resize image to have a height close to target_height pixels with slight random augmentation.

    Args:
        image (np.ndarray): Input image as a numpy array.
        target_height (int): Target height for the resized image. Default is 100.
        augment_factor (float): Factor to determine the range of random size augmentation. Default is 0.1 (±10%).

    Returns:
        np.ndarray: Resized and augmented image as a numpy array.
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h
    augmented_height = int(
        target_height * random.uniform(1 - augment_factor, 1 + augment_factor)
    )
    new_width = int(augmented_height * aspect_ratio)
    resized_image = np.array(
        Image.fromarray(image).resize((new_width, augmented_height))
    )
    return resized_image


def augment_image(
    image: np.ndarray,
    noise_prob: float = 0.5,
    brightness_prob: float = 0.5,
    contrast_prob: float = 0.5,
    blur_prob: float = 0.3,
    rotate_prob: float = 1.0,
    flip_prob: float = 0.0,
) -> np.ndarray:
    """
    Apply various augmentation techniques to the input image.

    Args:
        image (np.ndarray): Input image as a numpy array.
        noise_prob (float): Probability of adding noise. Default is 0.5.
        brightness_prob (float): Probability of adjusting brightness. Default is 0.5.
        contrast_prob (float): Probability of adjusting contrast. Default is 0.5.
        blur_prob (float): Probability of applying Gaussian blur. Default is 0.3.
        rotate_prob (float): Probability of rotating the image. Default is 0.3.
        flip_prob (float): Probability of flipping the image. Default is 0.3.

    Returns:
        np.ndarray: Augmented image as a numpy array.
    """

    def add_noise(img: np.ndarray, std_dev: float = 20.0) -> np.ndarray:
        noise = np.random.normal(0, std_dev, img.shape).astype(np.uint8)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
        return np.clip(img * factor, 0, 255).astype(np.uint8)

    def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

    def apply_gaussian_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR
        )

    # Apply augmentations probabilistically
    if random.random() < noise_prob:
        image = add_noise(image, std_dev=random.uniform(10, 30))

    if random.random() < brightness_prob:
        brightness_factor = random.uniform(0.7, 1.3)
        image = adjust_brightness(image, brightness_factor)

    if random.random() < contrast_prob:
        contrast_factor = random.uniform(0.7, 1.3)
        image = adjust_contrast(image, contrast_factor)

    if random.random() < blur_prob:
        kernel_size = random.choice([3, 5, 7])
        image = apply_gaussian_blur(image, kernel_size)

    if random.random() < rotate_prob:
        angle = random.uniform(-30, 30)
        image = rotate_image(image, angle)

    if random.random() < flip_prob:
        image = cv2.flip(image, 1)  # 1 for horizontal flip

    return image


def ocr_single_image(
    ocr_engine: PaddleOCR,
    image: np.ndarray,
) -> List[Tuple[List[List[float]], Tuple[str, float]]]:
    """Perform OCR on a single image."""
    result = ocr_engine.ocr(image, cls=False, det=False, rec=True)
    return result[0] if result else []


def ocr(
    config: Dict,
    image_file_paths: List[Path],
    language: str = "en",
    use_gpu: bool = False,
    use_angle_cls: bool = True,
    bootstrap_iterations: int = 9,
) -> Dict[str, List[List[Tuple[List[List[float]], Tuple[str, float]]]]]:
    """
    Perform bootstrapped OCR on the given images using PaddleOCR.

    Args:
        config (Dict): Configuration dictionary.
        image_file_paths (List[Path]): List of paths to input images.
        language (str): Language model to use. Default is 'en' for English.
        use_gpu (bool): Whether to use GPU for inference. Default is False.
        use_angle_cls (bool): Whether to use angle classification. Default is True.
        bootstrap_iterations (int): Number of bootstrap iterations. Default is 5.

    Returns:
        Dict[str, List[List[Tuple[List[List[float]], Tuple[str, float]]]]]: Bootstrapped OCR results.
    """
    ocr_engine = PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=language,
        use_gpu=use_gpu,
        ocr_version="PP-OCRv4",
        show_log=False,
    )
    all_results = {}
    for image_path in tqdm(image_file_paths, desc="Processing images"):
        try:
            logger.info(f"Processing image: {image_path}")
            # Load and resize image
            image = np.array(Image.open(image_path))
            image = resize_image(image, target_height=100, augment_factor=0.1)
            # Perform bootstrapped OCR
            bootstrap_results = []
            for _ in range(bootstrap_iterations):
                augmented_image = augment_image(image)
                result = ocr_single_image(ocr_engine, augmented_image)
                bootstrap_results.append(result)
            all_results[str(image_path)] = bootstrap_results
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
    return all_results
