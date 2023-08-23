# Classes
# Game: Data + Video for a Game
# Data: (One Data File for Game)
# Video: One Video File for a Game

# Utilites
# temporal_alignment
# spatial alignment

import os
import shutil

from numpy import full

from video import Video
from data import Data
from utilities.temporal_alignment import extract_timestamps

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
        self.create_game_folder()
        self.temporal_alignment()
        self.spatial_alignment()

    def create_game_folder(self):
        new_folder_path = f"videos-plus-data/{self.title}.{self.network}"
        if not os.path.exists(new_folder_path):
            try:
                os.mkdir(new_folder_path)
            except:
                print(f"Error creating folder with path {new_folder_path}.")
                raise Exception

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
        self.video.normalize()

    def temporal_alignment(self):
        extract_timestamps(self)

    def spatial_alignment(self):
        pass
