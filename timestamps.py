import json
import os
from video import Video
from data import Data

from utilities.timestamps.timestamp_extraction import extract_timestamps
from utilities.timestamps.timestamp_post_processing import postprocess_timestamps
from utilities.timestamps.timestamp_visualization import viz_timestamp_mapping
from utilities.files import File


class Timestamps:
    """Represents an instance of the extracted timestamps from a game."""

    def __init__(self, video: Video, data: Data, path=None) -> None:
        if path:
            self.path = os.path.abspath(path)
        else:
            self.path = path
        self.video = video
        self.data = data

    def get_path(self) -> str:
        """Path getter."""

        assert self.path
        return self.path

    def get_timestamps_data(self):
        """Raw timestamps .json data getter."""

        return File.load_json(self.get_path())

    def get_timestamps_quarter_time_map(self):
        """Different formating of timestamp data for moment-timestamp mapping."""

        raw_data = self.get_timestamps_data()
        return {
            f"{str(timestamp[0])} {str(timestamp[1])}": frame
            for frame, timestamp in raw_data.items()
        }

    def extract_timestamps(self):
        """Extract timestamps from a video. Create a file called timestamps.json in game folder."""

        assert self.path and not os.path.exists(
            self.path), f"Error: timestamp obj path is none or timestamps already exist at {self.path}."
        timestamps = json.dumps(extract_timestamps(
            self.video.path, self.video.network))
        assert os.path.exists(
            self.video.path), f"Error: video no longer exists at path {self.video.path}!"
        try:
            with open(self.path, "w") as outfile:
                outfile.write(timestamps)
        except:
            raise Exception(
                f"Error: could not save extracted timestamps to {self.path}.")

        post_processed_timestamps = postprocess_timestamps(self.path)
        post_processed_timestamps_path = self.path.rstrip(
            '.json') + '_modified.json'
        assert not os.path.exists(
            post_processed_timestamps_path), f"Error: post-processed timestamps already exists at {post_processed_timestamps_path}."
        with open(post_processed_timestamps_path, 'w') as file:
            json.dump(post_processed_timestamps, file, indent=4)

        # Replace original timestamps with post-processed results
        File.replace_path(os.path.abspath(self.path),
                          os.path.abspath(path=post_processed_timestamps_path))
        return self.get_path()
