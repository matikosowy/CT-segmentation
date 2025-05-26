"""
Train Module (train.py)
-------------------
This module contains the training loop for the 2D U-Net model used for kidney segmentation in CT images.
It includes functions for training, validation, and visualization of the training process.
"""

import gc
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from torch.optim import Adam
from monai.losses import DiceCELoss
from sklearn.metrics import jaccard_score
from torch.amp import GradScaler, autocast

from ctseg.eval import plot_history, evaluate_model
from ctseg.models import create_unet_model, create_segresnet_model
from ctseg.data import create_2d_segmentation_dataloaders, create_3d_segmentation_dataloaders


def train_2d_model(
    model,
    epochs,
    train_loader,
    val_loader,
    device,
    lr=1e-4,
    run_dir=datetime.now().strftime("%Y%m%d_%H%M%S"),
    weight_decay=1e-4,
    early_stopping_patience=8,
    optimizer=None,
    scheduler=None,
    start_epoch=0,
    best_dice=0.0,
    history=None,
):
    """
    Train the 2D U-Net model for kidney segmentation.
    This function handles the training loop, validation, and saving of the best model based on the Dice score.

    Args:
        model (nn.Module): Model to be trained.
        epochs (int): Number of training epochs.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the model on (CPU or GPU).
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        run_dir (str, Path, optional): Directory to save the model and training logs. Defaults to current date and time.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-5.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping training.
            Defaults to 8.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to Adam.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to ReduceLROnPlateau.
        start_epoch (int, optional): Epoch to start training from. Defaults to 0.
        best_dice (float, optional): Best Dice score achieved so far. Defaults to 0.0.
        history (dict, optional): Dictionary to store training and validation metrics. Defaults to None.

    Returns:
        model (nn.Module): Trained model.
        history (dict): Dictionary containing training and validation loss, Dice score, and Jaccard index.
    """

    loss_fn = DiceCELoss(
        to_onehot_y=False,
        sigmoid=True,
        include_background=True,
        lambda_ce=0.5,
    )

    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = GradScaler()

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_dice = 0.0 if best_dice is None else best_dice

    if history is None:
        train_losses, val_losses = [], []
        dice_scores, jaccard_scores = [], []
    else:
        train_losses = history["train_loss"]
        val_losses = history["val_loss"]
        dice_scores = history["dice_score"]
        jaccard_scores = history["jaccard_score"]

    early_stopping_counter = 0

    print("=" * 30)
    print("TRAINING")
    print("First epoch may take a bit longer... (0% for a while is normal)")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {start_epoch+epoch+1}/{start_epoch+epochs} [Train]")

        for batch in progress_bar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item(): .4f}"})

        model.eval()
        epoch_val_loss = 0.0
        epoch_dice, epoch_jaccard = 0.0, 0.0

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {start_epoch+epoch+1}/{start_epoch+epochs} [Val]")

            for batch in val_progress:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                masks_np = masks.cpu().numpy()

                outputs = model(images)
                loss = loss_fn(outputs, masks)

                epoch_val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                preds_np = (preds > 0.5).float().cpu().numpy()

                batch_dice = 0.0
                batch_jaccard = 0.0
                batch_count = 0

                for sample_idx in range(len(images)):
                    sample_dice = 0.0
                    sample_jaccard = 0.0
                    organ_count = 0

                    if masks_np[sample_idx].ndim > 2:
                        num_organs = masks_np[sample_idx].shape[0]

                        # For each organ channel
                        for organ_idx in range(num_organs):
                            pred = preds_np[sample_idx, organ_idx]
                            mask = masks_np[sample_idx, organ_idx]

                            # Only calculate metrics if this organ appears in this slice
                            if mask.sum() > 0:
                                dice_val = 2 * (pred * mask).sum() / (pred.sum() + mask.sum() + 1e-8)
                                sample_dice += dice_val

                                jac_val = jaccard_score(mask.flatten(), pred.flatten(), zero_division=1)
                                sample_jaccard += jac_val

                                organ_count += 1
                    else:
                        # Single channel mask (single organ)
                        pred = preds_np[sample_idx, 0] if preds_np[sample_idx].ndim > 2 else preds_np[sample_idx]
                        mask = masks_np[sample_idx]

                        if mask.sum() > 0:
                            dice_val = 2 * (pred * mask).sum() / (pred.sum() + mask.sum() + 1e-8)
                            sample_dice += dice_val

                            jac_val = jaccard_score(mask.flatten(), pred.flatten(), zero_division=1)
                            sample_jaccard += jac_val

                            organ_count += 1

                    # Average metrics across all organs for this sample
                    if organ_count > 0:
                        batch_dice += sample_dice / organ_count
                        batch_jaccard += sample_jaccard / organ_count
                        batch_count += 1

                # Average metrics across all samples in this batch
                if batch_count > 0:
                    epoch_dice += batch_dice / batch_count
                    epoch_jaccard += batch_jaccard / batch_count

        n_train_batches = max(1, len(train_loader))
        n_val_batches = max(1, len(val_loader))
        avg_train_loss = epoch_train_loss / n_train_batches
        avg_val_loss = epoch_val_loss / n_val_batches
        avg_dice = epoch_dice / n_val_batches
        avg_jaccard = epoch_jaccard / n_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        dice_scores.append(avg_dice)
        jaccard_scores.append(avg_jaccard)

        print(f"\nEpoch {start_epoch+epoch+1}/{start_epoch+epochs} summary: ")
        print(f"LR: {optimizer.param_groups[0]['lr']: .6f}")
        print(f"Train Loss: {avg_train_loss: .4f}")
        print(f"Val Loss: {avg_val_loss: .4f}")
        print(f"Dice Score: {avg_dice: .4f}")
        print(f"Jaccard Index: {avg_jaccard: .4f}")
        print("-" * 50)

        if avg_dice > best_dice:
            best_dice = avg_dice

            checkpoint = {
                "epoch": start_epoch + epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "history": {
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                    "dice_score": dice_scores,
                    "jaccard_score": jaccard_scores,
                },
            }

            torch.save(checkpoint, run_dir / "best_model.pth")
            print(f"New best model saved (Dice: {best_dice: .4f})\n")

            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered!\n({early_stopping_patience} epochs without improvement)\n")
                break

        scheduler.step(avg_val_loss)
        if optimizer.param_groups[0]["lr"] < 1e-6:
            print("LR too low, stopping training!\n")
            break

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "dice_score": dice_scores,
        "jaccard_score": jaccard_scores,
    }

    print("Training completed!")
    return model, history


def resume_training_2d(
    checkpoint,
    model,
    epochs,
    train_loader,
    val_loader,
    device,
    lr=1e-4,
    run_dir=datetime.now().strftime("%Y%m%d_%H%M%S"),
    weight_decay=1e-5,
):
    """
    Resume training from a checkpoint.
    This function loads the model and optimizer state from a checkpoint file and continues training.

    Args:
        checkpoint (str): Path to the checkpoint file.
        model (nn.Module): Model's architecture.
        epochs (int): Number of training epochs.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the model on (CPU or GPU).
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        run_dir (str, Path, optional): Directory to save the model and training logs. Defaults to current date and time.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-5.

    Returns:
        model (nn.Module): Trained model.
        history (dict): Dictionary containing training and validation loss, Dice score, and Jaccard index.
    """

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_dice = checkpoint["best_dice"]
    history = checkpoint["history"]

    model, history = train_2d_model(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        run_dir=run_dir,
        weight_decay=weight_decay,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_dice=best_dice,
        history=history,
    )

    return model, history


def train(args, device):
    """Main training wrapper for 2D segmentation.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        device (torch.device): Device to run the model on.
    """
    train_loader, val_loader, test_loader = create_2d_segmentation_dataloaders(
        root_dir=args.dataset,
        batch_size=args.batch_size,
        num_patients=args.num_patients,
        min_organ_pixels=args.min_organ_pixels,
        target_organs=args.target_organs,
        reset_cache=args.reset_cache,
    )

    if args.model == "unet":
        model = create_unet_model(
            in_channels=1,
            out_channels=len(args.target_organs),
            device=device,
            channels=args.unet_channels,
            dropout=args.unet_dropout,
            num_res_units=args.unet_units,
        )
    elif args.model == "segresnet":
        model = create_segresnet_model(
            in_channels=1,
            out_channels=len(args.target_organs),
            device=device,
            init_filters=args.segresnet_filters,
        )

    run_name = f"{args.model}{args.mode}_" + args.run_name
    run_dir = Path("runs") / Path(run_name)
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

    # Clean up after eval
    torch.cuda.empty_cache()
    gc.collect()


def train_3d_model():
    ...


def resume_training_3d():
    ...


def train_3d(args, device):
    """Main training wrapper for 3D segmentation.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        device (torch.device): Device to run the model on.
    """
    train_loader, val_loader, test_loader = create_3d_segmentation_dataloaders(
        root_dir=args.dataset,
        batch_size=args.batch_size,
        num_patients=args.num_patients,
        min_organ_pixels=args.min_organ_pixels,
        target_organs=args.target_organs,
        reset_cache=args.reset_cache,
        height=args.height,
    )

    if args.model == "unet":
        model = create_unet_model(
            dims=3,  # dims? in/out channels?
            in_channels=1,
            out_channels=len(args.target_organs),
            device=device,
            channels=args.unet_channels,
            dropout=args.unet_dropout,
            num_res_units=args.unet_units,
        )
    elif args.model == "segresnet":
        model = create_segresnet_model(
            dims=3,  # dims? in/out channels?
            in_channels=1,
            out_channels=len(args.target_organs),
            device=device,
            init_filters=args.segresnet_filters,
        )

    run_name = f"{args.model}{args.mode}_" + args.run_name
    run_dir = Path("runs") / Path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        print("=" * 50)
        print(f"Resuming training from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        torch.save(checkpoint["model_state_dict"], run_dir / "best_model.pth")

        model, history = resume_training_3d(
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
        model, history = train_3d_model(
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
        mode="3d",
        height=args.height,
    )

    # Clean up after eval
    torch.cuda.empty_cache()
    gc.collect()
