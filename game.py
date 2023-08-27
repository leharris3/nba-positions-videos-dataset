import os
import shutil
from utilities.timestamps.timestamp_visualization import viz_timestamp_mapping
from utilities.timestamps.timestamp_mapping import map_timestamps_to_statvu

from video import Video
from data import Data
from timestamps import Timestamps
from utilities.files import File


class Game:
    """Class representing an NBA basketball game, a player position data object, and a corroposonding video object."""

    def __init__(self, data: Data, video: Video) -> None:
        self.data = data
        self.video = video
        self.title = video.title
        self.network = video.network

    def get_folder_path(self) -> str:
        folder_path = f"videos-plus-data/{self.video.title}.{self.video.network}"
        return folder_path

    @classmethod
    def __init_with_title__(cls, title: str, network: str) -> object:
        """Init a game object with data from the statvu-raw-data folder."""

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

        print(f"Creating game folder at {self.get_folder_path()}.")
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
                assert isinstance(temp_data_path, str)
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

        print(f"Normalizing video at {self.video.path}.")
        if not self.video.is_normalized():
            normalized_video_path = self.video.normalize(preset="medium")
            File.replace_path(
                self.video.path, os.path.abspath(normalized_video_path))
            assert os.path.exists(self.video.path)

    def temporal_alignment(self) -> None:
        """Extract video timestamps and add to statvu data."""

        print(f"Add temporal information to game at {self.get_folder_path()}.")
        timestamps = Timestamps(self.video, self.data, os.path.abspath(
            f"videos-plus-data/{self.title}.{self.network}/timestamps.json"))
        timestamps_path = timestamps.extract_timestamps()
        trimmed_timestamps_path = self.video.trim_video_from_timestamps(
            timestamps_path)

        assert False, "Break"
        File.replace_path(timestamps.get_path(), trimmed_timestamps_path)
        map_timestamps_to_statvu(timestamps, self.data)

    def spatial_alignment(self) -> None:
        pass

    def visualize_timestamp_extraction(self):
        viz_timestamp_mapping(self.video, self.data, self.get_folder_path())
