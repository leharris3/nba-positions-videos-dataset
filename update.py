import os
import json
import logging

from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


annotations_dir = "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset/filtered-clip-annotations-bu"
all_annotation_fps = glob(annotations_dir + "/*/*/*.json")
root_dir = "/mnt/arc/levlevi/nba-positions-videos-dataset/nba-plus-statvu-dataset"


def get_updated_fps(annotation_fp: str):
    idx = annotation_fp.split("_")[-1][0]
    new_fp = (
        "/"
        + "/".join(annotation_fp.split("_annotation")[0].split("/")[-4:]).replace(
            "filtered-clip-annotations", "filtered-clips"
        )
        + f"_{idx}.mp4"
    )
    # full path to corresponding video for a given annotation
    full_path = root_dir + new_fp
    assert os.path.isfile(full_path), f"{full_path} does not exist"
    return new_fp, full_path


failed_json_write_cnt = 0
failed_json_write_list = []
failed_path_cnt = 0
failed_path_list = []
failed_open_cnt = 0
failed_open_list = []


def process_fp(fp: str):
    logger.info(f"processing {fp}")
    try:
        with open(fp, "r") as f:
            data = json.load(f)
    except Exception:
        logger.info(f"Failed to load annotation: {fp}")
        failed_open_cnt += 1
        failed_open_list.append(fp)
        return
    try:
        new_fp, full_path = get_updated_fps(fp)
        assert os.path.isfile(full_path), f"{full_path}"
        data["video_path"] = new_fp
        try:
            # overwrite the original annotation
            with open(fp, "w") as f:
                json.dump(data, f, indent=4)
        except Exception:
            logger.info(f"Failed to write annotation: {fp}")
            failed_json_write_cnt += 1
            failed_json_write_list.append(fp)
            return
    except Exception:
        logger.info(f"Failed to update annotation: {fp}")
        failed_path_cnt += 1
        failed_path_list.append(fp)
    logger.info(f"successfully updated annotation: {fp}")


with ProcessPoolExecutor(max_workers=64) as pool:
    for fp in tqdm(all_annotation_fps):
        pool.submit(process_fp, fp)

with open("failed_path.txt", "w") as f:
    f.write(failed_path_list)

with open("failed_cnt.txt", "w") as f:
    f.write(failed_open_list)

with open("failed_json_write.txt", "w") as f:
    f.write(failed_json_write_list)
