"""
Main script for CT segmentation project.
"""

import torch
import argparse
import gc
from pathlib import Path
from datetime import datetime

from ctseg.data import create_kidney_dataloaders
from ctseg.train import train_unet_2d
from ctseg.models import create_unet_2d_model

def debug_dataloader(dataloader, num_batches=2):
    print("\n=== Debug DataLoader ===")
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        images = batch['image']
        masks = batch['mask']
        print(f"Batch {i+1}:")
        print(f"  Image shape: {images.shape}, range: {images.min().item():.3f} to {images.max().item():.3f}")
        print(f"  Mask shape: {masks.shape}, range: {masks.min().item():.3f} to {masks.max().item():.3f}")
        print(f"  Mask unique values: {torch.unique(masks).tolist()}")
        print(f"  Mask channel sums: {masks.sum(dim=(0,2,3))}")

def main():
    """Main function to run the segmentation pipeline."""
    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser(description="CT Segmentation Training Pipeline")
    parser.add_argument("--mode", type=str, default="2d", choices=["2d", "3d"], 
                        help="Training mode: 2d or 3d")
    parser.add_argument("--dataset", type=str, default="dataset", 
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
                        help="Run name for saving model and logs")
    parser.add_argument("--num_patients", type=int, default=100,
                    help="Number of patients to use for training")
    parser.add_argument("--min_mask_pixels", type=int, default=50,
                    help="Number of patients to use for training")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Preparing data...")
    
    if args.mode == "2d":
        train_loader, val_loader, test_loader = create_kidney_dataloaders(
            root_dir=args.dataset,
            batch_size=args.batch_size,
            num_patients=args.num_patients,
            min_kidney_mask_pixels=args.min_mask_pixels,
        )
        
        #debug_dataloader(train_loader, num_batches=2)
        #debug_dataloader(val_loader, num_batches=2)

        print("Creating 2D UNet model...")
        out_channels = 1
        model = create_unet_2d_model(in_channels=1, out_channels=out_channels, device=device)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        print("Starting training...")
        
        run_dir = "runs" / Path(args.run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        model, history = train_unet_2d(
            model=model,
            epochs=args.epochs, 
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            run_dir=run_dir,
        )
            
        print(f"Best model saved at: {run_dir / 'best_model.pth'}")
        
        # todo: eval on test set, save results, plots from history
        print("Evaluating on test set...")
        
    else:  # mode == "3d"
        pass
    
    print("Training completed!")

if __name__ == "__main__":
    main()