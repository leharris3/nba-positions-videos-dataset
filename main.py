from temporal_grounding.temporal_grounding import *


def main():
    video_path = r"C:\Users\Levi\Desktop\localized-clock-roi-ds-raw-files\trimmed-videos\broadcast_2_period_1.mp4"
    save_path = r"data/example_timestamps.json"
    extract_timestamps_from_video(video_path=video_path, save_path=save_path)


if __name__ == "__main__":
    main()
