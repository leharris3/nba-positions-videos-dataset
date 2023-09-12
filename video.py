import cv2
import os


class Video:
    """Class representing an instance of a video."""

    def __init__(self, path: str) -> None:
        assert os.path.exists(path), f"Error: bad path to video."
        self.path = path

    def get_path(self) -> str:
        return self.path

    def set_path(self, path: str) -> None:
        assert os.path.exists(path), f"Error: path not found."
        self.path = path

    def get_video_attributes(self):
        """Getter method for basic video attributes."""

        cap = cv2.VideoCapture(self.get_path())
        if not cap.isOpened():
            raise Exception(
                f"Error: Unable to open video at path: {self.get_path()}.")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_attributes = {
            "width": frame_width,
            "height": frame_height,
            "fps": fps,
            "frame_count": frame_count
        }
        return video_attributes

    def move(self, to: str) -> None:
        pass
