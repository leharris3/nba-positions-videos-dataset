from shot_clip_extraction import process_dir


def main():
    dir_path = r"testing\shot_retrevial\B-100-T"
    shots_dir = r"testing\shot_retrevial\B-100-T-shot-clips"
    timestamps_dir = r"testing\shot_retrevial\B-100-T-timestamps"
    process_dir(dir_path, timestamps_dir, shots_dir, True)


if __name__ == "__main__":
    main()
