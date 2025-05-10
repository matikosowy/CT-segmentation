"""
Main script for CT segmentation project.
"""

import gc

import torch

from ctseg.train import train
from ctseg.eval import inference
from ctseg.args import parse_args


def main():
    """Main function to run the segmentation pipeline."""
    # Clean before starting
    torch.cuda.empty_cache()
    gc.collect()

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    if args.mode == "2d":
        if args.inference:
            inference(args, device)
        else:
            train(args, device)

    else:  # mode == "3d"
        pass


if __name__ == "__main__":
    main()
