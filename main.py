"""
Main script for CT segmentation project.
"""

import gc
import argparse
from pathlib import Path
from datetime import datetime

import torch

from ctseg.eval import plot_history, evaluate_model
from ctseg.data import create_2d_segmentation_dataloaders
from ctseg.train import train_2d_model, resume_training_2d
from ctseg.models import create_unet_2d_model, create_segresnet_2d_model


def main():
    """Main function to run the segmentation pipeline."""
    torch.cuda.empty_cache()
    gc.collect()

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
        "--min_organ_pixels", nargs="+", type=int, default=[50, 50, 30, 30], help="Minimum pixels for each organ mask"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    if args.mode == "2d":
        if args.inference:
            print("RUNNING INFERENCE ONLY...")
            assert args.checkpoint is not None, "Checkpoint path must be provided for inference!"
            model = create_segresnet_2d_model(
                in_channels=1,
                out_channels=len(args.target_organs),
                device=device,
            )
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            test_loader = create_2d_segmentation_dataloaders(
                root_dir=args.dataset,
                batch_size=args.batch_size,
                num_patients=args.num_patients,
                min_organ_pixels=args.min_organ_pixels,
                target_organs=args.target_organs,
                split="test",
                reset_cache=args.reset_cache,
            )

            # Save predictions in the same directory as the checkpoint
            checkpoint_path = Path(args.checkpoint)
            output_dir = checkpoint_path.parent / "eval"

            evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir=output_dir,
                organ_names=args.target_organs,
            )

            return

        train_loader, val_loader, test_loader = create_2d_segmentation_dataloaders(
            root_dir=args.dataset,
            batch_size=args.batch_size,
            num_patients=args.num_patients,
            min_organ_pixels=args.min_organ_pixels,
            target_organs=args.target_organs,
            reset_cache=args.reset_cache,
        )

        if args.model == "unet":
            model = create_unet_2d_model(
                in_channels=1,
                out_channels=len(args.target_organs),
                device=device,
            )
        elif args.model == "segresnet":
            model = create_segresnet_2d_model(
                in_channels=1,
                out_channels=len(args.target_organs),
                device=device,
            )

        run_name = f"{args.model}{args.mode}_" + args.run_name
        run_dir = "runs" / Path(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.resume:
            print("=" * 50)
            print(f"Resuming training from {args.checkpoint}...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            torch.save(checkpoint["model_state_dict"], run_dir / "best_model.pth")

            model, history = resume_training_2d(
                model=model,
                checkpoint=checkpoint,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                run_dir=run_dir,
                weight_decay=args.weight_decay,
            )

        else:
            print("=" * 50)
            print("Training model from scratch...")
            model, history = train_2d_model(
                model=model,
                epochs=args.epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                run_dir=run_dir,
                weight_decay=args.weight_decay,
            )

        # Clean up after training
        torch.cuda.empty_cache()
        gc.collect()

        plot_history(
            history=history,
            run_dir=run_dir,
        )

        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=run_dir / "eval",
            organ_names=args.target_organs,
        )

    else:  # mode == "3d"
        pass

    # Clean up after all operations
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
