"""
Main script for CT segmentation project.
"""

# todo:
# - train.py 3d support
# - eval.py 3d support
# - data.py (monai augmentations)

import gc

import torch

from ctseg.args import parse_args
from ctseg.train import train, train_3d
from ctseg.eval import inference, inference_3d


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
        if args.inference:
            inference_3d(args, device)
        else:
            train_3d(args, device)


if __name__ == "__main__":
    main()
