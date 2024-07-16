import logging
import json

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class NBAClockDataset(Dataset):

    def __init__(self, args):
        annotations_fp = args.annotations_fp
        logging.info(f"Loading annotations from: {annotations_fp}")
        with open(annotations_fp, "r") as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation = self.annotations[image_path]
        image = Image.open(image_path).convert("RGB")
        prompt = "<OCR>"
        time_on_clock = annotation.get("time_on_clock")
        if time_on_clock is not None:
            ground_truth_text = self.format_nba_clock_time(float(time_on_clock))
        else:
            ground_truth_text = "N/A"
        label = f"{{'<OCR>': '{ground_truth_text}'}}"
        return prompt, label, image

    @staticmethod
    def format_nba_clock_time(seconds) -> str:
        """
        Convert a float number of seconds to a string formatted as an NBA clock time.

        Args:
        seconds (float): Time remaining in seconds.

        Returns:
        str: Formatted time string (MM:SS, M:SS, SS.DS, or S.DS)
        """
        if seconds is None:
            return "N/A"
        if seconds < 0:
            raise ValueError("Time cannot be negative")
        minutes, seconds = divmod(seconds, 60)
        if minutes >= 1:
            # Format as MM:SS or M:SS
            return f"{int(minutes):01d}:{int(seconds):02d}"
        else:
            # Format as SS.DS or S.DS
            return f"{seconds:.1f}"


def prepare_dataloader(args, dataset: NBAClockDataset):
    return DataLoader(
        dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset)
    )
