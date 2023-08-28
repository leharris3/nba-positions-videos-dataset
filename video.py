import json
import os
import subprocess
import cv2

from utilities.files import File
from utilities.timestamps.timestamp_constants import *
from utilities.progress import print_progress


class Video:
    """Instance of a game video."""

    def __init__(self, path: str) -> None:
        assert os.path.exists(
            path), f"Error creating video: invalid valid path at {path}."
        self.path = os.path.abspath(path)
        try:
            self.title = path[-29: -8]
            self.network = path[-7: -4]
        except:
            raise Exception(f"Error: bad title format for video at {path}.")

    def get_path(self) -> str:
        assert self.path and os.path.exists(self.path)
        return self.path

    def normalize(self, preset: str) -> str:
        """Normalize a video given a path and save."""

        try:
            directory = os.path.dirname(self.path)
            file_name = os.path.splitext(os.path.basename(self.path))[0]
            temp_path = os.path.join(directory, f"{file_name}_converted.mp4")
            assert not os.path.exists(temp_path)
            command = [
                "ffmpeg",
                "-i", self.path,
                "-vf", "scale=1280:720",
                "-r", "25",
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", preset,
                "-c:a", "aac",
                "-b:a", "0",
                "-loglevel", "info",
                temp_path
            ]
            subprocess.run(command)
            return temp_path
        except:
            raise Exception(f"Failed to normalize video at {self.path}.")

    def is_normalized(self) -> bool:
        """Helper function. Returns true/false if video is already normalized."""

        try:
            cap = cv2.VideoCapture(self.path)
        except:
            raise Exception(f"Error: bad path to video: {self.path}.")
        if not cap.isOpened():
            return False
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        return frame_width == 1280 and frame_height == 720 and fps == 25

    def trim_video_from_timestamps(self, timestamps_path) -> tuple[str, str]:
        """Return path to a trimmed video. Iterate through timestamps and cut out 
        frames which do not have a vlid timestamp value"""

        print("Trimming frames without temporal information.")

        capture = cv2.VideoCapture(self.get_path())
        total_frames, output_width, output_height, output_fps = map(int, (
            capture.get(cv2.CAP_PROP_FRAME_COUNT),
            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            capture.get(cv2.CAP_PROP_FPS)
        ))

        output_codec = "mp4v"
        trimmed_video_path = f"{self.get_path().strip('.mp4')}_trim.mp4"
        video_writer = cv2.VideoWriter(
            trimmed_video_path, cv2.VideoWriter_fourcc(*output_codec), output_fps, (output_width, output_height))
        video_writer.set(cv2.CAP_PROP_BITRATE, TRIM_VIDEO_BITRATE)

        assert os.path.exists(
            timestamps_path), f"Error: bad path to timestamps {timestamps_path}."
        timestamps = File.load_json(timestamps_path)

        total_frames = len(timestamps.keys())
        new_frame_index = 0
        new_timestamps = {}
        for old_frame_index in timestamps:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(old_frame_index))
            res, frame = capture.read()
            if not res:
                raise Exception(
                    f"Error: could not read frame in from video at {self.get_path()}.")
            video_writer.write(frame)
            new_timestamps[new_frame_index] = timestamps[old_frame_index]
            new_frame_index += 1
            print_progress(new_frame_index / total_frames)

        video_writer.release()
        capture.release()
        trimmed_timestamps_path = f"{self.get_path().strip('.mp4')}_trimmed.json"
        File.save_json(new_timestamps, trimmed_timestamps_path)
        return trimmed_video_path, trimmed_timestamps_path
