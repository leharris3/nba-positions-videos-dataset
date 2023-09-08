import os
import cv2
import random
import shutil

# TODO: Make sure this shit works it prolly doesn't lol


def organize_lables(label_path: str):
    if not os.path.isdir(label_path):
        print(f"{label_path} is not a valid directory.")
        return

    for entry in os.listdir(label_path):
        entry_path = os.path.join(label_path, entry)
        if os.path.isdir(entry_path):
            split_data(entry_path)


def split_data(folder_path: str):

    video_path = None
    txts_path = f"{folder_path}/obj_train_data"
    if not os.path.isdir(txts_path):
        raise Exception("Invalid path.")

    for entry in os.listdir(folder_path):
        if ".mp4" in entry:
            video_path = entry
            break
    if not os.path.exists(video_path):
        raise Exception("Invalid path to video.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Couldn't open the video file.")

    train_percentage = 0.80
    val_percentage = 0.05
    frame_count = 0

    class_folder = None
    train_folder = "train/images"
    val_folder = "valid/images"
    test_folder = "test/images"

    # save first 5000 video frames to subdirs
    while frame_count < 5000:
        ret, frame = cap.read()
        if not ret:
            break

        rand = random.random()
        if rand < train_percentage:
            class_folder = train_folder
        elif rand < train_percentage + val_percentage:
            class_folder = val_folder
        else:
            class_folder = test_folder

        pad = (6 - len(str(frame_count))) * "0"
        data_title = f"frame_{pad}{frame_count}.txt"
        data_src = f"{folder_path}/obj_train_data/{data_title}"
        data_dst = f"{class_folder.strip('/images')}/labels/{data_title}"

        frame_filename = os.path.join(
            class_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        shutil.copy(data_src, data_dst)
        frame_count += 1

    cap.release()


organize_lables("raw-labels")
