from game import Game
from data import Data
from video import Video


def main():

    path_to_video = r"/content/drive/MyDrive/Research/sportvu-plus-video/unprocessed-videos/01.14.2016.LAL.at.GSW.TNT.mp4"
    path_to_data = r"/content/drive/MyDrive/Research/sportvu-plus-video/statvu-raw-data/01.14.2016.LAL.at.GSW.7z"
    data = Data(path_to_data)
    video = Video(path_to_video)
    game = Game(data, video)
    game.process()


main()
