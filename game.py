# Classes
# Game: Data + Video for a Game
# Data: (One Data File for Game)
# Video: One Video File for a Game

# Utilites
# temporal_alignment
# spatial alignment

from curses.panel import new_panel
import os
import shutil
import json

from numpy import full

from video import Video
from data import Data
from utilities.timestamp_extraction import extract_timestamps_plus_trim
from utilities.timestamp_post_processing import postprocess_timestamps

EXAMPLE_PATH = "/content/drive/MyDrive/Research/sportvu-plus-videos/statvu-plus-plus/01.14.2016.LAL.at.GSW.TNT/01.14.2016.LAL.at.GSW.TNT.mp4"


class Game:
    """Class representing an NBA basketball game, a player position data object, and a corroposonding video object."""

    def __init__(self, data: Data, video: Video) -> None:
        self.data = data
        self.video = video
        self.title = video.title
        self.network = video.network

    @classmethod
    def __init_with_title__(cls, title: str, network: str) -> object:
        full_title = f"{title}.{network}"
        path_to_data = f"statvu-raw-data/{full_title}.7z"
        path_to_video = f"unprocessed-videos/{full_title}.mp4"
        return Game(Data(path_to_data), Video(path_to_video))

    def process(self) -> None:
        """Perform temporal and spatial alignment for an NBA basketball game."""

        self.create_game_folder()
        normalized_video_path = self.video.normalize()
        self.replace_path(self.video.path, normalized_video_path)
        self.temporal_alignment()
        self.spatial_alignment()

    def create_game_folder(self) -> None:
        """Create a new folder for modified statvu data and video."""

        new_folder_path = f"videos-plus-data/{self.title}.{self.network}"
        if not os.path.exists(new_folder_path):
            try:
                os.mkdir(new_folder_path)
            except:
                print(f"Error creating folder with path {new_folder_path}.")
                raise Exception
        else:
            print(f"Error: folder already exists ar {new_folder_path}.")

        # copy data to new dir
        if not self.data.is_zipped:
            self.data.unzip(to=new_folder_path)
        else:
            new_data_path = f"{new_folder_path}/{self.title}.{self.network}.json"
            shutil.copy(self.data.path, new_data_path)
            self.data.path = os.path.abspath(new_data_path)

        # copy video to new dir
        new_video_path = f"{new_folder_path}/{self.title}.{self.network}.mp4"
        shutil.copy(self.video.path, new_video_path)
        self.video.path = os.path.abspath(new_video_path)

    def temporal_alignment(self) -> None:
        """Extract video timestamps and add to statvu data."""

        timestamps_path = f"videos-plus-data/{self.title}.{self.network}/timestamps.json"
        timestamps = json.dumps(extract_timestamps_plus_trim(
            self.video.path, self.network), indent=4)
        try:
            with open(timestamps_path, "w") as outfile:
                outfile.write(timestamps)
        except:
            print(
                f"Error: could now save extracted timestamps to {timestamps_path}.")
            raise Exception
        modified_timestamps_path = postprocess_timestamps(timestamps_path)

        self.replace_path(timestamps_path, modified_timestamps_path)

    def spatial_alignment(self) -> None:
        pass

    @classmethod
    def replace_path(cls, old_path: str, new_path: str) -> None:
        """Replace the file at old_path with file at new path. Rename new_path to old_path."""

        try:
            os.remove(old_path)
            os.rename(new_path, old_path)
        except:
            print(f"Error attemping to replace {old_path} with {new_path}.")
