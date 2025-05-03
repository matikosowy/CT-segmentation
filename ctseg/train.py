"""
Train Module (train.py)
-------------------
This module contains the training loop for the 2D U-Net model used for kidney segmentation in CT images.
It includes functions for training, validation, and visualization of the training process.
"""

from monai.losses import DiceCELoss
from torch.optim import Adam
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_score

from torch.amp import autocast, GradScaler


def train_unet_2d(
    model,
    epochs,
    train_loader,
    val_loader,
    device,
    lr=1e-4,
    save_path="unet2d_model.pth",
    weight_decay=1e-5
):
    
    loss_fn = DiceCELoss(
        to_onehot_y=False,  
        sigmoid=True,  
        squared_pred=False,  
        include_background=True,
        lambda_ce=0.5,  
    )
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_dice = 0.0
    train_losses, val_losses = [], []
    dice_scores, jaccard_scores = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Add channel dimension if needed
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        epoch_val_loss = 0.0
        epoch_dice, epoch_jaccard = 0.0, 0.0
   
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            for batch in val_progress:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                masks_np = masks.cpu().numpy()
                
                # Add channel dimension if needed
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                        
                epoch_val_loss += loss.item()
                
                preds = torch.sigmoid(outputs)
                preds_np = (preds > 0.5).float().cpu().numpy()
                
                dice = 0.0
                jaccard = 0.0
                count = 0
    
                for pred, mask in zip(preds_np, masks_np):
                    # Ensure pred and mask are 2D
                    if pred.ndim > 2:
                        pred = pred.squeeze()
                        
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                     
                    if mask.sum() > 0:
                        dice_val = 2 * (pred * mask).sum() / (pred.sum() + mask.sum() + 1e-8)
                        dice += dice_val
                        jac_val = jaccard_score(mask.flatten(), pred.flatten())
                        jaccard += jac_val
                        count += 1
                
                if count > 0:
                    epoch_dice += dice / count
                    epoch_jaccard += jaccard / count
                    
        n_train_batches = max(1, len(train_loader))
        n_val_batches = max(1, len(val_loader))
        avg_train_loss = epoch_train_loss / n_train_batches
        avg_val_loss = epoch_val_loss / n_val_batches
        avg_dice = epoch_dice / n_val_batches
        avg_jaccard = epoch_jaccard / n_val_batches
        
        scheduler.step(avg_val_loss)

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), save_path)
            print(f"\nNew best model saved (Dice: {best_dice:.4f})")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        dice_scores.append(avg_dice)
        jaccard_scores.append(avg_jaccard)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Dice Score: {avg_dice:.4f}")
        print(f"Jaccard Index: {avg_jaccard:.4f}")
        print("-" * 50)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dice_scores, label='Dice Score')
    plt.plot(jaccard_scores, label='Jaccard Index')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'dice_score': dice_scores,
        'jaccard_score': jaccard_scores
    }

    return model, history