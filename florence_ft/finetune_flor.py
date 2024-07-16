# adopted from: https://huggingface.co/blog/finetune-florence2

# paper details
# batch-size: 2048
# unfrozen vison encoder

# hugging face ft config
# 8xH100 (80GB)
# batch-size: 64
# full-ft

# our config
# 8xA6000 (48GB)
# batch-size: 32
# full-ft
# lr: 1e-6

# TODO:
# 1. multi-gpu support
# 2. slurm script

# multiprocess example: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
import torch
import torch.distributed
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import json
import os
import logging

from typing import Dict
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AdamW
from utils.logging import setup_primary_logging, setup_worker_logging
from utils.parser import get_parser
from utils.dataset import NBAClockDataset
from utils.models import get_model_and_processor


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    setup_worker_logging(rank, log_queue)
    logging.info(f"Initializing process group: rank={rank}, world_size={world_size}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        checkpoint_dir = "/playpen-storage/levlevi/nba-positions-videos-dataset/florence_ft/results"
        PATH = os.path.join(checkpoint_dir, f"model_{epoch}.pth")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(args):
    train_set = NBAClockDataset(args)
    model, _ = get_model_and_processor(args)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        no_deprecation_warning=True,
    )
    return train_set, model, optimizer


def prepare_dataloader(args, dataset: NBAClockDataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset)
    )

def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(args)
    train_data = prepare_dataloader(args, dataset)
    trainer = Trainer(model, train_data, optimizer, rank, args.save_every)
    trainer.train(args.epochs)
    destroy_process_group()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    global log_queue
    log_queue = setup_primary_logging("out.log", "error.log")
    torch.multiprocessing.set_start_method("spawn", force=True)
    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
