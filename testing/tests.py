import os
import numpy as np
from PIL import Image
from utilities.text_extraction.timestamp_extraction import *


def test_ocr_benchmark():
    """Tests text-extraction on a set of benchmark images. Returns an accuracy score as a [0-1] float."""

    path_to_test_images = r"demos/example_rois"
    images = [path for path in os.listdir(path_to_test_images)]

    found, total = 0, 0
    for path in images:
        img_path = os.path.join(path_to_test_images, path)
        image = Image.open(img_path)
        img_arr = np.array(image)
        result = extract_timestamps_from_image(img_arr, print_results=True)
        if result.time_remaining and result.quarter:
            found += 1
        total += 1

    return found / total
