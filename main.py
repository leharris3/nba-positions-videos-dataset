import os

from shot_clip_extraction import *
from viz import *


def main():
    choose_ten_games_with_logs()


if __name__ == "__main__":
    main()

# Video -> Process -> Shot Clips

# 1. find 100 videos with matching logs, move to a "to_upload" dir
# 2. upload 100 videos to folder A in gdrive
# 3. extract timestamps from 100 videos on gdrive to folder B
# 4. extract video clips from 100 videos to folder C?
# 5. remove 100 videos from folder A
# 6. move 100 videos from "to_upload" to a "processed dir"
