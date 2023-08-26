import os
import subprocess
import cv2

from timestamps import Timestamps


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

    def trim_video_from_timestamps(self, timestamps: Timestamps) -> str:
        """Return path to a trimmed video. Iterate through timestamps and cut out 
        frames which do not have a vlid timestamp value"""

        capture = cv2.VideoCapture(self.get_path())
        total_frames, output_width, output_height, output_fps = map(int, (
            capture.get(cv2.CAP_PROP_FRAME_COUNT),
            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            capture.get(cv2.CAP_PROP_FPS)
        ))

        output_codec = "mp4v"
        output_path = f"{self.get_path().strip('.mp4')}_trim.mp4"
        video_writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*output_codec), output_fps, (output_width, output_height))

        # video_writer.write(frame)
        # video_writer.release()
        # TODO: finish function
        return output_path
