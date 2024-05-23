import json

from pipeline.pipeline import process_dir


def main():

    timestamps = process_dir(
        "/mnt/opr/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/test-set/",
        "/mnt/opr/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/dummy",
    )
    with open("timestamps.json", "w") as f:
        json.dump(timestamps, f)


if __name__ == "__main__":
    main()
