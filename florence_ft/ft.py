# stolen from: https://github.com/andimarafioti/florence2-finetuning
# roboflow tutorial: https://blog.roboflow.com/fine-tune-florence-2-object-detection/

import yaml
import argparse
import os
import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import wandb
import warnings
import numpy as np
import sys

from PIL import Image
from tqdm import tqdm
from functools import partial

from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler
from transformers import logging as transformers_logging

CHECKPOINT = "andito/Florence-2-large-ft"

wandb.login(key="3d8c09b359c1abc995fd03c27398c41afce857c1")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.optimization"
)
warnings.filterwarnings(
    "ignore",
    message="Grad strides do not match bucket view strides",
    module="torch.autograd.graph",
)
transformers_logging.set_verbosity_error()


class NBAClockDataset(Dataset):

    def __init__(self, config, split):
        if split == 'train':
            annotations_fp = config["train_annotations_fp"]
        elif split == 'val':
            annotations_fp = config["val_annotations_fp"]
        else:
            raise Exception(f"Error: invalid split: {split}")
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
            try:
                ground_truth_text = self.format_nba_clock_time(time_on_clock)
            except ValueError as e:
                print(f"Error formatting time: {e}")
                ground_truth_text = "ERROR"
        else:
            ground_truth_text = "N/A"
        label = f"<OCR>: {ground_truth_text}"
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
        # already in correct format
        if type(seconds) == str:
            return seconds
        if seconds is None:
            return "N/A"
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError("Time cannot be negative")
        minutes, seconds = divmod(seconds, 60)
        if minutes >= 1:
            # Format as MM:SS or M:SS
            return f"{int(minutes):01d}:{int(seconds):02d}"
        else:
            # Format as SS.DS or S.DS
            return f"{seconds:.1f}"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        # max_length=800,
    ).to(device)
    return inputs, answers


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=val_sampler,
    )
    return train_loader, val_loader


def train_model(
    rank,
    config,
    world_size,
    batch_size=6,
    epochs=10,
    lr=1e-6,
    eval_steps=10,
    run_name=None,
):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # Initialize wandb
    if rank == 0:  # Only initialize wandb in the main process
        wandb.init(project="Clock OCR FT", name=run_name)
        wandb.config.update(
            {
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "eval_steps": eval_steps,
                "world_size": world_size,
            }
        )

    # train dataset
    train_dataset = NBAClockDataset(config, split="train")
    
    # val dataset
    val_dataset = NBAClockDataset(config, split="val")
    
    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, trust_remote_code=True,).to(device)
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT, trust_remote_code=True,)
    
    model = DDP(model, device_ids=[rank])

    # Create DataLoaders
    num_workers = 0
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers = batch
            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1

        # Log training loss to wandb
        avg_train_loss = train_loss / len(train_loader)
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss})

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                inputs, answers = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                val_loss += loss.item()
                
        # Log training loss to wandb
        avg_eval_loss = val_loss / len(val_loader)
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_eval_loss})
                
        # Save model checkpoint
        if rank == 0:  # Only the main process saves the checkpoint
            output_dir = f"./model_checkpoints/{run_name}/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    # Finish the wandb run
    if rank == 0:
        wandb.finish()

    cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune florence-2 on Clock OCR dataset"
    )
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--batch-size", type=int, default=6, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=1000,
        help="Number of steps between evaluations",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(
            config,
            world_size,
            args.batch_size,
            args.epochs,
            args.lr,
            args.eval_steps,
            args.run_name,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
