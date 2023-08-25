import os
import subprocess
import cv2


class Video:
    """Instance of a game video."""

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"Error creating video: invalid valid path at {path}.")
            raise Exception
        self.path = os.path.abspath(path)
        try:
            self.title = path[-29: -8]
            self.network = path[-7: -4]
        except:
            print(f"Error creating video with path {path}.")
            raise Exception

    # TODO: change preset -> slow for better compression
    def normalize(self) -> str:
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
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "0",
                "-loglevel", "info",
                temp_path
            ]
            subprocess.run(command)
            return temp_path
        except:
            print(f"Failed to normalize video at {self.path}.")
            raise Exception

    def is_normalized(self):
        try:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                return False
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = int(cap.get(5))
            if frame_width == 1280 and frame_height == 720 and fps == 25:
                return True
        except:
            return False
