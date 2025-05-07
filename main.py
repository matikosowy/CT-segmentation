"""
Main script for CT segmentation project.
"""

import torch
import argparse
import gc
from pathlib import Path
from datetime import datetime

from ctseg.data import create_kidney_dataloaders
from ctseg.train import train_unet_2d, resume_training_2d
from ctseg.models import create_unet_2d_model, create_segresnet_2d_model
from ctseg.viz import plot_history
from ctseg.eval import evaluate_model

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
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
                        help="Run name for saving model and logs")
    parser.add_argument("--num_patients", type=int, default=100,
                    help="Number of patients to use for training")
    parser.add_argument("--min_mask_pixels", type=int, default=50,
                    help="Number of patients to use for training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory for resuming training")
    parser.add_argument("--reset_cache", action="store_true",
                        help="Reset the dataset file cache")
    parser.add_argument("--inference", action="store_true",
                        help="Only run evaluation on test set using saved model")
    parser.add_argument("--model", type=str, default="segresnet",
                        choices=["unet", "segresnet"],
                        help="Model architecture to use: unet or segresnet")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    if args.mode == "2d":
        # if args.reset_cache:
        #     print("Resetting dataset cache...")
        #     reset_dataset_cache()
        
        if args.inference:
            print("RUNNING INFERENCE ONLY...")
            assert args.checkpoint is not None, "Checkpoint path must be provided for inference!"
            model = create_segresnet_2d_model(
                in_channels=1, 
                out_channels=1, 
                device=device,
            )
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_loader = create_kidney_dataloaders(
                root_dir=args.dataset,
                batch_size=args.batch_size,
                num_patients=args.num_patients,
                min_kidney_mask_pixels=args.min_mask_pixels,
                mode="test",
            )
            
            evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir="inference_results",
            )
            
            return
        
        train_loader, val_loader, test_loader = create_kidney_dataloaders(
            root_dir=args.dataset,
            batch_size=args.batch_size,
            num_patients=args.num_patients,
            min_kidney_mask_pixels=args.min_mask_pixels,
        )
        
        out_channels = 1
        model = create_segresnet_2d_model(
            in_channels=1, 
            out_channels=out_channels, 
            device=device,
        )
        
        run_name = f"{args.model}_" + args.run_name
        run_dir = "runs" / Path(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if args.resume:
            print("="*50)
            print(f"Resuming training from {args.checkpoint}...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model, history = resume_training_2d(
                model=model,
                checkpoint=checkpoint,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs, 
                lr=args.lr,
                run_dir=run_dir,
            )
            
        else:
            print("="*50)
            print("Training model from scratch...")
            model, history = train_unet_2d(
                model=model,
                epochs=args.epochs, 
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                run_dir=run_dir,
            )
        
        plot_history(
            history=history,
            run_dir=run_dir,
        )
          
        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=run_dir/"eval",
        )
        
    else:  # mode == "3d"
        pass
    

if __name__ == "__main__":
    main()