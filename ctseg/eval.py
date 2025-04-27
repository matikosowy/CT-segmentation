import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from monai.inferers import sliding_window_inference
from monai.transforms import DivisiblePad
from matplotlib.widgets import Slider
from matplotlib import colors as mcolors


def inference_on_new_data(model, image_path, device, output_path='segmentation_result.nii.gz'):
    """
    Perform inference on a new CT image.
    
    Args:
        model (nn.Module): Trained model.
        image_path (str): Path to the CT image (NIFTI format).
        device (torch.device): Computation device.
        output_path (str): Path to save the segmentation result.
    
    Returns:
        tuple: Original image and segmentation result as numpy arrays.
    """
    image_data = nib.load(image_path).get_fdata()
    original_shape = image_data.shape
    
    image_data = np.expand_dims(image_data, axis=(0, 1))  # Add batch and channel dims
    image_tensor = torch.from_numpy(image_data.astype(np.float32)).to(device)
    
    affine = nib.load(image_path).affine
    
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    
    # Remove batch dimension for padding
    image_tensor = image_tensor.squeeze(0)
    
    # Pad for network
    padder = DivisiblePad(k=16)
    image_tensor_padded = padder(image_tensor)
    
    # Add batch dimension back
    image_tensor_padded = image_tensor_padded.unsqueeze(0)
    
    print(f"Original shape: {image_tensor.shape}, Padded shape: {image_tensor_padded.shape}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(
            image_tensor_padded, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model, overlap=0.5
        )
        pred = output.argmax(dim=1).cpu().numpy().squeeze()
    
    # Remove padding
    pred = pred[
        :original_shape[0],
        :original_shape[1],
        :original_shape[2]
    ]
    
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), output_path)
    print(f"Segmentation saved to {output_path}")
    
    return image_data.squeeze(), pred


def evaluate_model(model, val_loader, device, output_dir="evaluation_results"):
    """
    Evaluate model performance on the validation set and save visualizations.
    
    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Computation device.
        output_dir (str): Directory to save results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
   
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if i >= 3:  # Evaluate only first 3 samples
                break
            
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            # Make prediction with sliding window
            outputs = sliding_window_inference(
                inputs, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model, overlap=0.5
            )
            
            # Convert to numpy arrays
            inputs_np = inputs.cpu().numpy()[0, 0]  # [B, C, H, W, D] -> [H, W, D]
            labels_np = labels.cpu().numpy()[0, 0]  # [B, C, H, W, D] -> [H, W, D]
            outputs_np = outputs.argmax(dim=1).cpu().numpy()[0]  # [B, C, H, W, D] -> [H, W, D]
            
            # Save results
            for z in range(0, inputs_np.shape[2], inputs_np.shape[2]//5):  # Sample slices
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(inputs_np[:, :, z], cmap='gray')
                plt.title('Input CT')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(inputs_np[:, :, z], cmap='gray')
                plt.imshow(labels_np[:, :, z], cmap='jet', alpha=0.3)
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(inputs_np[:, :, z], cmap='gray')
                plt.imshow(outputs_np[:, :, z], cmap='jet', alpha=0.3)
                plt.title('Prediction')
                plt.axis('off')
                
                plt.savefig(f'{output_dir}/sample_{i}_slice_{z}.png')
                plt.close()
                

def browse_ct_segmentation_colored(ct_volume, segmentation_volume):
    """
    Interactive CT visualization with segmentation overlay and a slider.
    
    Args:
        ct_volume (np.ndarray): CT volume data.
        segmentation_volume (np.ndarray): Segmentation mask data.
    """
    if ct_volume.ndim > 3:
        while ct_volume.ndim > 3:
            if ct_volume.shape[0] == 1:
                ct_volume = ct_volume[0]
            else:
                break
    
    if segmentation_volume.ndim > 3:
        while segmentation_volume.ndim > 3:
            if segmentation_volume.shape[0] == 1:
                segmentation_volume = segmentation_volume[0]
            else:
                break
    
    print(f"Wizualizacja - kształt CT: {ct_volume.shape}, kształt segmentacji: {segmentation_volume.shape}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    slice_idx = ct_volume.shape[-1] // 2

    cmap = mcolors.ListedColormap(['none', 'red', 'blue'])
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)

    ct_img = ax.imshow(ct_volume[:, :, slice_idx], cmap="gray")
    seg_img = ax.imshow(segmentation_volume[:, :, slice_idx], cmap=cmap, norm=norm, alpha=0.4)
    ax.set_title(f'Przekrój {slice_idx}/{ct_volume.shape[-1]-1}')
    ax.axis('off')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="gray", lw=1),
        plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.4),
        plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.4)
    ]
    ax.legend(legend_elements, ['Tło', 'Prawa nerka', 'Lewa nerka'], 
              loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False)


    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, 'Przekrój', 0, ct_volume.shape[-1] - 1, valinit=slice_idx, valfmt='%0.0f')


    def update(val):
        idx = int(slider.val)
        ct_img.set_data(ct_volume[:, :, idx])
        seg_img.set_data(segmentation_volume[:, :, idx])
        ax.set_title(f'Przekrój {idx}/{ct_volume.shape[-1]-1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()