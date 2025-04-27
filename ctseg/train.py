from monai.losses import DiceCELoss, DiceFocalLoss
from torch.optim import Adam
from monai.inferers import sliding_window_inference
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def train_unet(model, epochs, train_loader, val_loader, device, 
               lr=3e-4, save_path="unet3d_model.pth", weight_decay=1e-5):
    """
    Train a 3D UNet model with advanced training techniques.

    Args:
        model (torch.nn.Module): U-Net model.
        epochs (int): Number of epochs to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device (CPU or CUDA).
        lr (float): Learning rate.
        save_path (str): Path to save the trained model.
        weight_decay (float): L2 regularization weight.
        
    Returns:
        dict: Dictionary containing training history.
    """

    class_weights = torch.tensor([0.1, 1.0, 1.0], device=device)
    loss_function = DiceFocalLoss(
        include_background=True,
        to_onehot_y=True, 
        gamma=2.0,
        lambda_dice=0.8,
        lambda_focal=0.2,
        weight=class_weights,
    )
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'lr': [],
        'best_epoch': 0,
        'best_dice': 0.0
    }

    # Early stopping
    best_val_dice = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Trening
        model.train()
        running_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for batch_data in train_loop:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Eval
        model.eval()
        val_running_loss = 0.0
        val_dice_metric = 0.0
        val_count = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Valid]", leave=False)
        with torch.no_grad():
            for batch_data in val_loop:
                val_inputs = batch_data["image"].to(device)
                val_labels = batch_data["label"].to(device)
                
                # Use sliding window inference for validation
                roi_size = (128, 128, 128)
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, 1, model, overlap=0.5
                )
                
                val_loss = loss_function(val_outputs, val_labels)
                val_running_loss += val_loss.item()
                
                # Calculate Dice score
                val_outputs = torch.softmax(val_outputs, dim=1)
                y_pred = torch.argmax(val_outputs, dim=1)
                y_true = val_labels.squeeze(1)
                
                for i in range(1, 3):  # Calculate Dice for each class (1=right kidney, 2=left kidney)
                    dice_score = calculate_dice(y_pred == i, y_true == i)
                    val_dice_metric += dice_score
                    val_count += 1
                
                val_loop.set_postfix(loss=val_loss.item())
        
        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_dice = val_dice_metric / val_count if val_count > 0 else 0.0
        
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)
        
        print(f" Epoch {epoch}/{epochs}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save model if best 
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            epoch_save_path = f"{model_dir}/unet3d_epoch{epoch}_dice{avg_val_dice:.4f}.pth"
            torch.save(model.state_dict(), epoch_save_path)
            torch.save(model.state_dict(), save_path)
            history['best_epoch'] = epoch
            history['best_dice'] = avg_val_dice
            print(f"💾 New best model saved! Dice: {avg_val_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
                break
        
        # Save training curves
        if epoch % 5 == 0 or epoch == epochs:
            plot_training_curves(history, f"{model_dir}/training_curves_epoch{epoch}.png")
    
    print(f" Training completed. Best model from epoch {history['best_epoch']} with Dice {history['best_dice']:.4f}")
    
    # Final plot
    plot_training_curves(history, f"{model_dir}/final_training_curves.png")
    
    return history

def calculate_dice(pred, target):
    """
    Calculate Dice coefficient between binary tensors.
    """
    intersection = torch.sum(pred & target).item()
    union = torch.sum(pred).item() + torch.sum(target).item()
    
    if union == 0:
        return 1.0  # If both pred and target are empty, Dice is 1
    
    return 2.0 * intersection / union

def plot_training_curves(history, save_path):
    """
    Plot and save training curves.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Dice score
    ax2.plot(history['val_dice'], label='Val Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate
    ax3.plot(history['lr'], label='Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()