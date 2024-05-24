import json
import sys
import os

from pipeline import process_dir


def main():

    test_set_fp = "/mnt/opr/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/assets/test-set"
    timestamps_out_fp = "/mnt/opr/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/assets/annotations/timestamps.json"
    timestamps = process_dir(test_set_fp)
    with open(timestamps_out_fp, "w") as f:
        json.dump(timestamps, f, indent=4)


if __name__ == "__main__":
    main()
