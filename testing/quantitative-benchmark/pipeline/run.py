import json

from pipeline import process_dir


def main():

    test_set_fp = "/mnt/sun/levlevi/data-sources"
    timestamps_out_fp = "/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/data/nba_15_16_timestamps.json"
    timestamps = process_dir(test_set_fp)
    with open(timestamps_out_fp, "w") as f:
        json.dump(timestamps, f, indent=4)


if __name__ == "__main__":
    main()
