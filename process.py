import os

from pipeline import *
from viz import *


def main():

    dir = r"C:\Users\Levi\Desktop\contextualized-shot-quality-estimation\testing\test-batches\B-100-T"
    data_path = r"C:\Users\Levi\Desktop\contextualized-shot-quality-estimation\testing\test-batches\B-100-T-Results\data"
    viz_path = r"C:\Users\Levi\Desktop\contextualized-shot-quality-estimation\testing\test-batches\B-100-T-Results\viz"

    # vids = os.listdir(dir)
    # for vid in vids:
    #     vid_path = os.path.join(dir, vid)
    #     viz_path = os.path.join(viz_dir, vid.replace('.avi', "_viz_.avi"))
    #     roi = extract_roi_from_video(vid_path)
    #     visualize_roi(vid_path, viz_path, roi)

    process_dir(dir, data_path, viz_path)


if __name__ == "__main__":
    main()
