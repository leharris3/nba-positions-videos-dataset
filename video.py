import os
import subprocess


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

    def normalize(self) -> str:
        """Normalize a video given a path and save."""

        try:
            directory = os.path.dirname(self.path)
            file_name = os.path.splitext(os.path.basename(self.path))[0]
            temp_path = os.path.join(directory, f"{file_name}_converted.mp4")
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
