from video import Video


class Game:
    """Instance of basketball game. Contains information about teams, date of play, etc."""

    def __init__(self, video: Video, data=None) -> None:
        self.video = video
