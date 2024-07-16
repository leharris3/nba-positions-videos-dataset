import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--annotations_fp", type=str, help="Path to the annotations file"
    )
    parser.add_argument("--save_every", type=int, help="How often to save a snapshot")
    parser.add_argument("--epochs", type=int, help="Total epochs to train the model")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="Number of distributed processes (default: 1)",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-6,
        type=float,
        help="The learning rate (default: 1e-6)",
    )
    return parser
