import cv2
import numpy as np
from PIL import Image
from copy import copy

SAVE_PATH = r"testing\demos\output_images"


def preprocess_image_for_tesseract(image, save=None):
    """Preprocess a ROI for OCR."""

    def change_dpi(image, target_dpi=95):
        """95 is the magic number, font height should be 30-33 px for best results."""

        try:
            image = Image.fromarray(image)
            current_dpi = image.info.get("dpi", (72, 72))
            scale_factor = target_dpi / current_dpi[0]
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            resized_image = image.resize((new_width, new_height))
            resized_image.info["dpi"] = (target_dpi, target_dpi)
            return np.array(resized_image)
        except Exception as e:
            raise Exception("An error while preprocessing a frame:", str(e))

    scaled_image = change_dpi(image)

    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)[1]
    # kernel = np.ones((1, 1), np.uint8)
    # result = cv2.dilate(thresh, kernel, iterations=2)
    # result = thresh

    # result_c1 = copy(result)
    # result_c2 = copy(result)

    # black_pixels = result_c1[np.where(result_c1 == 0)].size
    # white_pixels = result_c2[np.where(result_c2 == 255)].size

    # if black_pixels > white_pixels:
    #     result = cv2.bitwise_not(result)

    # if type(save) is bool and save:
    #     out_path = f"{SAVE_PATH}/pp.png"
    #     cv2.imwrite(out_path, result)

    return scaled_image


def nparr_from_img_path(img_path):
    """Return a numpy array from an img_path."""

    image = Image.open(img_path)
    return np.array(image)
