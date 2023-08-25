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

from video import Video
from data import Data
from utilities.files import File
from utilities.timestamp_extraction import extract_timestamps_plus_trim
from utilities.timestamp_post_processing import postprocess_timestamps
from utilities.timestamp_visualization import viz_extraction

EXAMPLE_PATH = "/content/drive/MyDrive/Research/sportvu-plus-videos/statvu-plus-plus/01.14.2016.LAL.at.GSW.TNT/01.14.2016.LAL.at.GSW.TNT.mp4"


class Game:
    """Class representing an NBA basketball game, a player position data object, and a corroposonding video object."""

    def __init__(self, data: Data, video: Video) -> None:
        self.data = data
        self.video = video
        self.title = video.title
        self.network = video.network
        self.folder_path = f"videos-plus-data/{video.title}.{video.network}"

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

        new_folder_path = f"videos-plus-data/{self.title}.{self.network}"
        if not os.path.exists(new_folder_path):
            try:
                os.mkdir(new_folder_path)
            except:
                print(f"Error creating folder with path {new_folder_path}.")
                raise Exception
        else:
            print(f"Folder already exists at {new_folder_path}.")

        # copy data to new dir, if data is zipped: unzip
        new_data_path = f"{new_folder_path}/{self.title}.{self.network}.json"

        if not os.path.exists(new_data_path):
            if self.data.is_zipped:
                temp_data_path = self.data.unzip(to=new_folder_path)
                os.rename(temp_data_path, new_data_path)
            else:
                shutil.copy(self.data.path, new_data_path)
            self.data.path = os.path.abspath(new_data_path)
            assert self.data.path == os.path.abspath(new_data_path)
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
            normalized_video_path = self.video.normalize()
            File.replace_path(self.video.path, normalized_video_path)
            assert os.path.exists(normalized_video_path)

    def temporal_alignment(self) -> None:
        """Extract video timestamps and add to statvu data."""

        timestamps_path = f"videos-plus-data/{self.title}.{self.network}/timestamps.json"
        try:
            assert not os.path.exists(timestamps_path)
        except:
            print("Error: extracted timestamps already exist. Please remove.")
            raise Exception
        timestamps = json.dumps(extract_timestamps_plus_trim(
            self.video.path, self.network), indent=4)
        # video exists after being trimmed + replaced
        assert os.path.exists(self.video.path)
        try:
            with open(timestamps_path, "w") as outfile:
                outfile.write(timestamps)
        except:
            print(
                f"Error: could not save extracted timestamps to {timestamps_path}.")
            raise Exception
        modified_timestamps_path = postprocess_timestamps(timestamps_path)

        raise Exception  # break
        File.replace_path(timestamps_path, modified_timestamps_path)

    def spatial_alignment(self) -> None:
        pass

    def visualize_timestamp_extraction(self):
        viz_extraction(self.video.path, self.data.path, f"")
        pass
