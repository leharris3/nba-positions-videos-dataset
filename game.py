# Classes
# Game: Data + Video for a Game
# Data: (One Data File for Game)
# Video: One Video File for a Game

# Utilites
# temporal_alignment
# spatial alignment

import os
import shutil
import json
from sqlite3 import Time
import time
from utilities.timestamp_visualization import viz_timestamp_mapping
from utilities.timestamp_mapping import map_timestamps_to_statvu

from video import Video
from data import Data
from timestamps import Timestamps
from utilities.files import File

EXAMPLE_PATH = "/content/drive/MyDrive/Research/sportvu-plus-videos/statvu-plus-plus/01.14.2016.LAL.at.GSW.TNT/01.14.2016.LAL.at.GSW.TNT.mp4"


class Game:
    """Class representing an NBA basketball game, a player position data object, and a corroposonding video object."""

    def __init__(self, data: Data, video: Video) -> None:
        self.data = data
        self.video = video
        self.title = video.title
        self.network = video.network

    def get_folder_path(self) -> str:
        return f"videos-plus-data/{self.video.title}.{self.video.network}"

    @classmethod
    def __init_with_title__(cls, title: str, network: str) -> object:
        full_title = f"{title}.{network}"
        path_to_data = f"statvu-raw-data/{title}.7z"
        path_to_video = f"unprocessed-videos/{full_title}.mp4"
        return Game(Data(path_to_data), Video(path_to_video))

    def process(self) -> None:
        """Perform temporal and spatial alignment for an NBA basketball game."""

        self.create_game_folder()
        self.normalize_video()
        self.temporal_alignment()
        self.spatial_alignment()

    def create_game_folder(self) -> None:
        """Create a new folder for modified statvu data and video."""

        new_folder_path = self.get_folder_path()
        if not os.path.exists(new_folder_path):
            try:
                os.mkdir(new_folder_path)
            except:
                raise Exception(
                    f"Error creating folder with path {new_folder_path}.")
        else:
            print(f"Folder already exists at {new_folder_path}.")

        # copy data to new dir, if data is zipped: unzip
        new_data_path = f"{new_folder_path}/{self.title}.{self.network}.json"

        if not os.path.exists(new_data_path):
            if self.data.is_zipped:
                temp_data_path = self.data.unzip(to=new_folder_path)
                assert temp_data_path is str
                os.rename(temp_data_path, new_data_path)
            else:
                shutil.copy(self.data.path, new_data_path)
            self.data.path = os.path.abspath(new_data_path)
            assert os.path.exists(new_data_path)
        else:
            print(f"Data file exists and formatted at: {new_data_path}.")

        # copy video to new dir
        new_video_path = f"{new_folder_path}/{self.title}.{self.network}.mp4"
        if not os.path.exists(new_video_path):
            shutil.copy(self.video.path, new_video_path)
            self.video.path = os.path.abspath(new_video_path)
            assert os.path.exists(new_video_path)
        else:
            print(
                f"Video already exists at {new_video_path}. Will normalize if necessary.")
            self.video.path = new_video_path
        assert self.video.path == new_video_path

    def normalize_video(self) -> None:
        """Normalize a video to dim 1280x720 w/ 25 FPS."""

        if not self.video.is_normalized():
            normalized_video_path = self.video.normalize(preset="ultrafast")
            File.replace_path(
                self.video.path, os.path.abspath(normalized_video_path))
            assert os.path.exists(normalized_video_path)

    def temporal_alignment(self) -> None:
        """Extract video timestamps and add to statvu data."""

        timestamps = Timestamps(self.video, self.data)
        timestamps.path = os.path.abspath(
            f"videos-plus-data/{self.title}.{self.network}/timestamps.json")
        assert not os.path.exists(
            timestamps.path), "Error: extracted timestamps already exist. Please remove."

        timestamps.extract_timestamps()
        map_timestamps_to_statvu(timestamps, self.data)

    def spatial_alignment(self) -> None:
        pass

    def visualize_timestamp_extraction(self):
        viz_timestamp_mapping(self.video, self.data, self.get_folder_path())
