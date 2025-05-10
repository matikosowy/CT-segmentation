import argparse
from datetime import datetime


def parse_args():
    """Parse command line arguments for the CT segmentation training pipeline.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="CT Segmentation Training Pipeline")

    parser.add_argument(
        "--mode",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="Training mode: 2d or 3d",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        help="Run name for saving model and logs",
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=100,
        help="Number of patients to use for training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory for resuming training",
    )
    parser.add_argument(
        "--reset_cache",
        action="store_true",
        help="Reset the dataset file cache",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Only run evaluation on test set using saved model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="segresnet",
        choices=["unet", "segresnet"],
        help="Model architecture to use: unet or segresnet",
    )
    parser.add_argument(
        "--target_organs",
        nargs="+",
        default=["kidney_right", "kidney_left", "adrenal_gland_right", "adrenal_gland_left"],
        help="List of target organs to segment",
    )
    parser.add_argument(
        "--min_organ_pixels",
        nargs="+",
        type=int,
        default=[50, 50, 30, 30],
        help="Minimum pixels for each organ mask",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer",
    )

    args = parser.parse_args()

    return args
