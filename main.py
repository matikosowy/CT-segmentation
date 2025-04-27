"""
Main script for CT segmentation project.
"""

import torch
import nibabel as nib

from ctseg.data import get_labels, prepare_data, prepare_totalsegmentator_dataset
from ctseg.train import train_unet
from ctseg.models import create_unet_model,  create_segresnet_model
from ctseg.eval import inference_on_new_data, evaluate_model, browse_ct_segmentation_colored


def main():
    """Main function to run the segmentation pipeline."""
    torch.cuda.empty_cache()
    
    dataset_path = "dataset"
    roi_subset = ["kidney_right", "kidney_left"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Preparing data...")
    train_loader, val_loader, train_files, val_files = prepare_totalsegmentator_dataset(
        dataset_path, 
        roi_subset=roi_subset
    )
    
    print("Creating model...")
    model = create_unet_model(in_channels=1, out_channels=3, device=device)
    
    print("Starting training...")
    history = train_unet(
        model=model,
        epochs=100, 
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=3e-4,
        save_path="unet_kidneys.pth"
    )
    
    model = model.load_state_dict(torch.load("unet_kidneys.pth"))
    

    # Inference on first validation example
    test_image_path = val_files[0]['image']
    print(f"Performing inference on {test_image_path}...")
    image_data_raw, segmentation_result = inference_on_new_data(
        model, 
        test_image_path, 
        device, 
        'segmented_totalsegmentator_result.nii.gz'
    )
    
    label_path = val_files[0]['label']
    label_data = nib.load(label_path).get_fdata()
    
    print("CT shape:", image_data_raw.shape)
    print("Label shape:", label_data.shape)
    print("Prediction shape:", segmentation_result.shape)
    
    print("Displaying results...")
    browse_ct_segmentation_colored(image_data_raw, segmentation_result)

if __name__ == "__main__":
    main()