from game import Game
from data import Data
from video import Video


def main():

    path_to_video = r"/content/drive/MyDrive/Research/sportvu-plus-video/videos-plus-data/01.14.2016.LAL.at.GSW.TNT/01.14.2016.LAL.at.GSW.TNT.mp4"
    path_to_data = r"/content/drive/MyDrive/Research/sportvu-plus-video/videos-plus-data/01.14.2016.LAL.at.GSW.TNT/01.14.2016.LAL.at.GSW.TNT.json"
    data = Data(path_to_data)
    video = Video(path_to_video)
    game = Game(data, video)
    game.temporal_alignment()


main()
