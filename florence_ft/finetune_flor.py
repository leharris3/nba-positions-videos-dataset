# adopted from: https://huggingface.co/blog/finetune-florence2

# paper details
# batch-size: 2048
# unfrozen vison encoder

# hugging face ft config
# 8xH100 (80GB)
# batch-size: 64
# full-ft

# our config
# 8xA6000 (48GB)
# batch-size: 32
# full-ft
# lr: 1e-6

import torch
import json
import os
import logging
import argparse
import numpy as np
import yaml

from typing import Dict
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer
from transformers import AdamW, get_scheduler


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NBAClockDataset(Dataset):

    _model = None
    _processor = None
    _config = None

    def __init__(self, config):
        annotations_fp = config["annotations_fp"]
        with open(annotations_fp, "r") as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())
        NBAClockDataset._model, NBAClockDataset._processor = get_model_and_processor(
            config
        )
        NBAClockDataset._config = config

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation = self.annotations[image_path]
        image = Image.open(image_path).convert("RGB")
        prompt = "<OCR>"
        ground_truth_text = NBAClockDataset.format_nba_clock_time(
            annotation["time_remaining"]
        )
        label = f"{'<OCR>': '{ground_truth_text}'}"
        return prompt, label, image

    @classmethod
    def format_nba_clock_time(seconds: float) -> str:
        """
        Convert a float number of seconds to a string formatted as an NBA clock time.

        Args:
        seconds (float): Time remaining in seconds.

        Returns:
        str: Formatted time string (MM:SS, M:SS, SS.DS, or S.DS)
        """
        if seconds < 0:
            raise ValueError("Time cannot be negative")
        minutes, seconds = divmod(seconds, 60)
        if minutes >= 1:
            # Format as MM:SS or M:SS
            return f"{int(minutes):01d}:{int(seconds):02d}"
        else:
            # Format as SS.DS or S.DS
            return f"{seconds:.1f}"

    @classmethod
    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = NBAClockDataset._processor(
            text=list(questions), images=list(images), return_tensors="pt", padding=True
        ).to(NBAClockDataset._config["device"])
        return inputs, answers


def get_model_and_processor(config: Dict):
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base-ft",
        trust_remote_code=True,
    ).to(config["device"])
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base-ft",
        trust_remote_code=True,
    )
    return model, processor


def main(config):
    # train dataset + dataloader
    train_dataset = NBAClockDataset(config["annotations_fp"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=train_dataset.collate_fn,
    )
    logger.info(f"Created dataset with {len(train_dataset)} samples")

    optimizer = AdamW(NBAClockDataset._model.parameters(), lr=config["learning_rate"])
    num_training_steps = config["num_epochs"] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=num_training_steps,
    )
    
    for epoch in range(config["num_epochs"]):
        break 
        NBAClockDataset._model.train() 
        train_loss = 0
        i = -1
        for inputs, answers in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{config["num_epochs"]}"):
            i += 1
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"] 
            labels = NBAClockDataset._processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(config["device"])
            outputs = NBAClockDataset._model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss}")

    NBAClockDataset._model.save_pretrained(config["model_output_dir"])
    NBAClockDataset._processor.save_pretrained(config["processor_output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Florence-2 model on NBA clock dataset"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
