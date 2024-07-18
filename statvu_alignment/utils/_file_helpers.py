import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_annotations(annotations_fp: str):
    with open(annotations_fp, "r") as f:
        annotations = json.load(f)

    replace_str = "C:/Users/Levi/Desktop/quantitative-benchmark/test-set\\"
    with_str = "/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/assets/test-set/"
    keys = list(annotations.keys())
    for k in keys:
        new_key = k.replace(replace_str, with_str)
        if new_key != k:
            annotations[new_key] = annotations[k]
            del annotations[k]

    return annotations


def get_timestamps(timestamps_fp: str):
    with open(timestamps_fp, "r") as f:
        timestamps = json.load(f)

    replace_str = "/playpen-storage/levlevi/"
    with_str = "/playpen-storage/"

    for k in list(timestamps.keys()):
        new_key = k.replace(replace_str, with_str)
        if new_key != k:
            timestamps[new_key] = timestamps[k]
            del timestamps[k]

    return timestamps
