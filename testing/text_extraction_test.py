import os
import numpy as np
import time

from PIL import Image
from unittest import TestCase
from utilities.text_extraction.timestamp_extraction import *
from utilities.text_extraction.preprocessing import *


class TextExtractionTests(TestCase):

    def test_extract_timestamps_from_image(self):
        assert True

    def test_ocr_benchmark(self):
        """Tests text-extraction on a set of benchmark images. Returns an accuracy score as a [0-1] float."""

        path_to_test_images = r"testing\assets\example_rois"
        images = [path for path in os.listdir(path_to_test_images)]

        start_time = time.time()
        found, total = 0, 0
        for path in images:
            img_path = os.path.join(path_to_test_images, path)
            image = Image.open(img_path)
            img_arr = np.array(image)
            result = extract_timestamps_from_image(
                img_arr, print_results=False, preprocessing_func=preprocess_image_for_tesseract, extraction_method="tesseract")
            if result.time_remaining and result.quarter:
                found += 1
            total += 1

        print("%s seconds" % ((time.time() - start_time) / total), " / image")
        print(found, "/",  total)

        assert total > 0
        assert (found / total) > .90
