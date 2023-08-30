# A Video Should Encapsulate:
# living path to video file
#

import os


class Video:
    """Class representing an instance of a video."""

    def __init__(self, path: str) -> None:
        assert os.path.exists(path), f"Error: bad path to video."
        self.path = path

    def get_path(self):
        return self.path

    def move(to: str) -> None:
        assert os.path.exists(to), f"Error: bad path {to}."

        pass
