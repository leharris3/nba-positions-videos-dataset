import os

from shot_clip_extraction import *
from viz import *


def main():
    print(find_first_game_with_log())


if __name__ == "__main__":
    main()

# 1. find 10 videos which have valid game logs
# 2. upload them to a gdrive folder
# 3. extract timestamps and save
# 4. use timestamps and game logs to find shot clips
