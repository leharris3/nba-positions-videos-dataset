import os
import random

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from glob import glob
from tqdm import tqdm
from trdg.generators import GeneratorFromStrings

# File paths
fonts_dir = "/mnt/arc/levlevi/nba-positions-videos-dataset/florence_ft/data/fine-tunning-set/fonts"
dictionary_txt_fp = (
    "/mnt/arc/levlevi/nba-positions-videos-dataset/florence_ft/data/nba_game_times.txt"
)
dst_dir = "/mnt/arc/levlevi/nba-positions-videos-dataset/florence_ft/data/fine-tunning-set/synth_data"

# Ensure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Create a generator with various augmentations
generator = GeneratorFromStrings(
    [string for string in open(dictionary_txt_fp, "r").readlines()],
    blur=random.randint(0, 3),
    random_blur=True,
    fonts=glob(fonts_dir + "/*.ttf"),
    background_type=0,  # Use image background
    size=random.randint(32, 64),  # Random font size between 32 and 64
    skewing_angle=random.randint(0, 5),  # Random skewing
    random_skew=True,
    distorsion_type=random.randint(0, 3),  # Random distortion type
    distorsion_orientation=random.randint(0, 2),  # Random distortion orientation
    is_handwritten=False,  # Set to True if you want handwritten-style text
    width=200 + random.randint(-100, 100),  # Random width
    alignment=1,  # Center alignment
    text_color="#000000,#FFFFFF",
    character_spacing=random.randint(0, 3),  # Random character spacing
    margins=(5, 5, 5, 5),  # Add some margins
    fit=True,  # Ensure text fits in the image
)


def save_img(i, img, lbl):
    # Create a filename for the image
    if img == None:
        return
    filename = f"game_clock_{i:04d}.jpg"
    filepath = os.path.join(dst_dir, filename)
    # Save the image
    img.save(filepath)
    # Save the label (text) to a separate file
    label_filename = f"game_clock_{i:04d}.txt"
    label_filepath = os.path.join(dst_dir, label_filename)
    with open(label_filepath, "w") as f:
        f.write(lbl)


with ProcessPoolExecutor(max_workers=64) as ex:
    for i, (img, lbl) in tqdm(enumerate(generator), total=100940):
        try:
            future = ex.submit(save_img, i, img, lbl)
            future.result()
        except Exception as e:
            print(e)


print(f"Generated {i+1} game clock images in {dst_dir}")
