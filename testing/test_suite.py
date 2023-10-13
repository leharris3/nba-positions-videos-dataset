import os
import numpy as np
import time

from PIL import Image
from unittest import TestCase

from temporal_grounding.temporal_grounding import *


class RegexTests(TestCase):

    def test_blank_regex(self):

        results = None
        actual = find_time_remaining_from_results(results)
        expected = None
        assert actual == expected, f"Expected {expected}, actual: {actual}."

        results = []
        actual = find_time_remaining_from_results(results)
        expected = None
        assert actual == expected, f"Expected {expected}, actual: {actual}."

    def test_regex_with_spaces(self):

        results = ["11:21 "]
        assert find_time_remaining_from_results(
            results) == "11:21", f"Expected 11:21, got: {find_time_remaining_from_results(results)}"

        results = ["A, 23237895", "3", "  0    8:   43   "]
        assert find_time_remaining_from_results(
            results) == "08:43", f"Expected 8:43, got: {find_time_remaining_from_results(results)}"

    def test_regex_no_match(self):

        results = ["Abc8:2as0r"]
        assert find_time_remaining_from_results(results) is None

        results = ["88:21"]
        assert find_time_remaining_from_results(results) is None

        results = ["0.21"]
        actual = find_time_remaining_from_results(results)
        expected = None
        assert actual == expected, f"Expected {expected}, actual: {actual}."

        results = ["31:21"]
        actual = find_time_remaining_from_results(results)
        expected = None
        assert actual == expected, f"Expected {expected}, actual: {actual}."

    def test_regex_edge_cases(self):

        results = ["20:00"]
        actual = find_time_remaining_from_results(results)
        expected = "20:00"
        assert actual == expected, f"Expected {expected}, actual: {actual}."

        results = ["080:23", "15:23.2", "2:30"]
        actual = find_time_remaining_from_results(results)
        expected = "15:23.2"
        assert actual == expected, f"Expected {expected}, actual: {actual}."


class TimeConversionTests(TestCase):

    def test_time_coversion_blank(self):

        time = None
        actual = convert_time_to_float(time)
        expected = None
        assert actual == expected, f"Expected {expected}, actual: {actual}."

        time = ""
        actual = convert_time_to_float(time)
        expected = None
        assert actual == expected, f"Expected {expected}, actual: {actual}."

    def test_time_coversion_standard(self):

        time = "1:23"
        actual = convert_time_to_float(time)
        expected = (60.0 * 1) + 23.0
        assert actual == expected, f"Expected {expected}, actual: {actual}."

        time = "08:45"
        actual = convert_time_to_float(time)
        expected = (60.0 * 8) + 45.0
        assert actual == expected, f"Expected {expected}, actual: {actual}."

        time = "50.2"
        actual = convert_time_to_float(time)
        expected = 50.2
        assert actual == expected, f"Expected {expected}, actual: {actual}."

    def test_time_rare_formatting(self):

        time = "09:45.2"
        actual = convert_time_to_float(time)
        expected = (60.0 * 9) + 45.0 + .2
        assert actual == expected, f"Expected {expected}, actual: {actual}."


class TextExtractionTests(TestCase):

    def test_extract_text_with_paddle_none(self):

        image = None
        results = extract_text_with_paddle(image)
        assert results == []

    def test_extract_text_with_paddle_valid(self):

        image = Image.open(
            r"testing\assets\example_cropped_rois\time_remaining\time_remaining_1.PNG")
        results = extract_text_with_paddle(image)
        assert results == ["2:41"]

    def test_extract_text_with_paddle_quarter(self):

        image = Image.open(
            r"testing\assets\example_cropped_rois\quarter\quarter_1.PNG")
        results = extract_text_with_paddle(image)
        assert results == ["4th"]

    def test_extract_time_remaining_from_image_none(self):

        image = None
        result = extract_time_remaining_from_image(image)
        assert result is None

    def test_extract_time_remaining_from_image_valid(self):

        image = Image.open(
            r"testing\assets\example_cropped_rois\time_remaining\time_remaining_1.PNG")
        result = extract_time_remaining_from_image(image)
        assert result == "2:41"


class TimeExtractionBenchmarkTests(TestCase):

    def whole_time_remaining_roi_benchmark(self):

        found, total = 0, 0
        dir = r"testing\assets\example_rois"
        rois = os.listdir(dir)
        roi_paths = [os.path.join(dir, roi) for roi in rois]
        for fp in roi_paths:
            img = Image.fromarray(cv2.imread(fp))
            result = extract_time_remaining_from_image(img)
            if result is not None:
                found += 1
            total += 1
        assert (found / total) >= .85

    def cropped_time_remaining_roi_benchmark(self):

        found, total = 0, 0
        dir = r"testing\assets\example_cropped_rois\time_remaining"
        rois = os.listdir(dir)
        roi_paths = [os.path.join(dir, roi) for roi in rois]
        for fp in roi_paths:
            img = Image.fromarray(cv2.imread(fp))
            result = extract_time_remaining_from_image(img)
            if result is not None:
                found += 1
            total += 1
        assert (found / total) >= .95


class PostProcessTests(TestCase):

    def test_post_process_timestamps_none(self):

        timestmaps = None
        frame_rate = 0.0
        try:
            post_process_timestamps(timestmaps, frame_rate)
            assert False
        except:
            assert True

    def test_post_process_timestamps_base(self):

        timestamps = {"0": {
            "quarter": 1,
            "time_remaining": 300.0
        }}
        expected = {"0": {
            "quarter": 1,
            "time_remaining": 300.0
        }}
        not_expected = {}
        frame_rate = 30.0

        post_process_timestamps(timestamps, frame_rate)
        try:
            assert type(timestamps) == dict
            assert timestamps == expected
            assert not timestamps == not_expected
        except:
            raise Exception(f"Expected {expected}, got {timestamps}.")

    def test_post_process_timestamps_basic_inter(self):

        timestamps = {"0": {
            "quarter": 1,
            "time_remaining": 300.0
        }, "1": {
            "quarter": 1,
            "time_remaining": None
        }, "2": {
            "quarter": 1,
            "time_remaining": 299.0
        }, "3": {
            "quarter": 1,
            "time_remaining": None
        }}

        expected = {"0": {
            "quarter": 1,
            "time_remaining": 300.0
        }, "1": {
            "quarter": 1,
            "time_remaining": 299.5
        }, "2": {
            "quarter": 1,
            "time_remaining": 299.0
        }, "3": {
            "quarter": 1,
            "time_remaining": 298.5
        }}

        not_expected = {}
        frame_rate = 2.0

        post_process_timestamps(timestamps, frame_rate)
        try:
            assert type(timestamps) == dict
            assert timestamps == expected
            assert not timestamps == not_expected
        except:
            raise Exception(f"Expected {expected}, got {timestamps}.")

    def test_post_process_timestamps_inter_hard(self):

        timestamps = {"0": {
            "quarter": 1,
            "time_remaining": 300.0
        }, "1": {
            "quarter": 1,
            "time_remaining": None
        }, "2": {
            "quarter": 1,
            "time_remaining": None
        }, "3": {
            "quarter": 1,
            "time_remaining": None
        }, "4": {
            "quarter": 1,
            "time_remaining": 300.0
        }, "5": {
            "quarter": 1,
            "time_remaining": 299.0
        }, "6": {
            "quarter": 1,
            "time_remaining": None
        }, "7": {
            "quarter": 1,
            "time_remaining": None
        }, "8": {
            "quarter": 1,
            "time_remaining": 298.0
        }}

        expected = {"0": {
            "quarter": 1,
            "time_remaining": 300.0
        }, "1": {
            "quarter": 1,
            "time_remaining": 300.0
        }, "2": {
            "quarter": 1,
            "time_remaining": 299.75
        }, "3": {
            "quarter": 1,
            "time_remaining": 299.5
        }, "4": {
            "quarter": 1,
            "time_remaining": 299.25
        }, "5": {
            "quarter": 1,
            "time_remaining": 299.0
        }, "6": {
            "quarter": 1,
            "time_remaining": 298.75
        }, "7": {
            "quarter": 1,
            "time_remaining": 298.5
        }, "8": {
            "quarter": 1,
            "time_remaining": 298.0
        }}

        {'0': {'quarter': 1, 'time_remaining': 300.0}, 
         '1': {'quarter': 1, 'time_remaining': 300.0}, 
         '2': {'quarter': 1, 'time_remaining': 299.8}, 
         '3': {'quarter': 1, 'time_remaining': 299.5}, 
         '4': {'quarter': 1, 'time_remaining': 299.2}, 
         '5': {'quarter': 1, 'time_remaining': 298.8}, 
         '6': {'quarter': 1, 'time_remaining': 298.5}, 
         '7': {'quarter': 1, 'time_remaining': 298.2}, 
         '8': {'quarter': 1, 'time_remaining': 297.8}}

        not_expected = {}
        frame_rate = 4.0

        post_process_timestamps(timestamps, frame_rate)
        try:
            assert type(timestamps) == dict
            assert timestamps == expected
            assert not timestamps == not_expected
        except:
            raise Exception(f"Expected {expected}, got {timestamps}.")
