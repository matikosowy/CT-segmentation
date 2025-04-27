"""
Data Module (data.py)
-------------------
This module handles the data preparation and segmentation mask generation for the CT segmentation task.

It includes the following functionalities:
1. **get_segmentation_masks**: This function generates segmentation masks for NIfTI files in the specified directory using the TotalSegmentator library.
2. **prepare_data**: This function prepares the data for training by applying various transformations and creating a DataLoader.
"""

from totalsegmentator.python_api import totalsegmentator
from monai.transforms import *
from monai.data import Dataset, DataLoader, list_data_collate
from pathlib import Path
from sklearn.model_selection import train_test_split
import tqdm
from monai.utils import first

from torch.utils.data import Dataset

import torch

class Kidneys2dDataset(Dataset):
    def __init__(self, root_dir, transform=None, combine_kidneys=True):
        self.samples = []
        self.transform = transform
        self.combine_kidneys = combine_kidneys
        
        patient_dirs = sorted([d for d in Path(root_dir).iterdir() if d.is_dir()])
        
        for patient_dir in patient_dirs:
            patient_path = Path(patient_dir)
            ct_path = patient_path / "ct.nii.gz"
            left_path = patient_path / "kidney_left.nii.gz"
            right_path = patient_path / "kidney_right.nii.gz"
            
            if not ct_path.exists() or not left_path.exists() or not right_path.exists():
                print(f"Missing files for {patient_dir}, skipping.")
                continue
        
            ct_volume = nib.load(ct_path).get_fdata()
            left_volume = nib.load(left_path).get_fdata()
            right_volume = nib.load(right_path).get_fdata()
            
            assert ct_volume.shape == left_volume.shape == right_volume.shape, "CT and mask volumes must have the same shape."
            
            depth = ct_volume.shape[2]
            
            for idx in range(depth):
                left_slice = left_volume[:, :, idx]
                right_slice = right_volume[:, :, idx]
                
                if np.sum(left_slice) == 0 and np.sum(right_slice) == 0:
                    continue # Skip empty slices
                
                self.samples.append({
                    'ct_slice': ct_volume[:, :, idx],
                    'left_mask': left_slice,
                    'right_mask': right_slice,
                })
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            image = sample['ct_slice']
            left = sample['left_mask']
            right = sample['right_mask']
            
            if self.combine_kidneys:
                mask = np.clip(left + right, 0, 1)[None, ...]
            else:
                mask = np.stack([left, right], axis=0)
            
            if self.transform:
                data = {'image': image, 'label': mask}
                data = self.transform(data)
                image = torch.tensor(data['image'], dtype=torch.float32)
                mask = torch.tensor(data['label'], dtype=torch.float32)

            return image, mask
        

def get_totalsegmentator_labels(input_dir, output_dir, roi_subset=None):
    """
    Generate segmentation masks for NIfTI files in the specified directory.
    Uses the TotalSegmentator library to perform the segmentation.
    
    Args:
        input_dir (str or Path): Directory containing NIfTI files.
        output_dir (str or Path): Directory to save the generated segmentation masks.
        roi_subset (list, optional): List of regions of interest to segment.
    """
    
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    for directory in input_dir.iterdir():
        if not directory.is_dir():
            continue
            
        print(f"Processing directory: {directory}")
        
        out_dir = output_dir / directory.name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in directory.glob('*.nii*'):
            if filename.suffix == '.nii' or filename.suffix == '.nii.gz':
                print(f"  Processing file: {filename.name}")
                
                output_path = out_dir / filename.stem
                output_path.mkdir(parents=True, exist_ok=True)
                
                # check if the mask already exists
                if output_path.exists():
                    print(f"  Mask already exists for {filename.name}, skipping.")
                    continue

                try:
                    totalsegmentator(
                        str(filename),
                        str(output_path),
                        roi_subset=roi_subset,
                    )
                    print(f"  Successfully segmented {filename.name}")
                except Exception as e:
                    print(f"  Error processing {filename.name}: {str(e)}")
                    
        print(f"Completed processing directory: {directory}\n")

    print("All processing complete!")
    
    
from pathlib import Path
import nibabel as nib
import numpy as np
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

def prepare_data(images_path, masks_path, roi_subset, val_split=0.2, seed=42):
    """
    Prepare the data for training by applying various transformations and creating DataLoaders.
    
    Args:
        images_path (str or Path): Path to the directory containing CT images
        masks_path (str or Path): Path to the directory containing mask folders.
        roi_subset (list of str): List of organs to use
        val_split (float): Validation split ratio (0.0-1.0)
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, train_files, val_files)
    """
    set_determinism(seed=seed)
    images_path = Path(images_path)
    masks_path = Path(masks_path)

    all_files = []
    skipped_count = 0
    total_count = 0

    for patient_dir in sorted(p.name for p in images_path.iterdir() if p.is_dir()):
        patient_path = images_path / patient_dir
        mask_patient_path = masks_path / patient_dir

        if not patient_path.is_dir():
            continue

        for image_file in sorted(f.name for f in patient_path.iterdir() if f.suffix in ['.nii', '.nii.gz']):
            total_count += 1
            img_full_path = patient_path / image_file
            mask_folder = mask_patient_path / Path(image_file).stem

            if not mask_folder.exists():
                print(f"Mask folder not found for {img_full_path}, skipping.")
                skipped_count += 1
                continue

            # Initialize empty mask
            first_mask_path = mask_folder / f"{roi_subset[0]}.nii.gz"
            if not first_mask_path.exists():
                print(f"No organ masks found for {img_full_path}, skipping.")
                skipped_count += 1
                continue

            example_mask = nib.load(first_mask_path)
            merged_mask_data = np.zeros(example_mask.shape, dtype=np.uint8)
            merged_mask_affine = example_mask.affine

            # Fill merged mask
            for idx, organ_name in enumerate(roi_subset, start=1):
                organ_mask_path = mask_folder / f"{organ_name}.nii.gz"
                if organ_mask_path.exists():
                    organ_mask = nib.load(organ_mask_path).get_fdata()
                    merged_mask_data[organ_mask > 0] = idx
                else:
                    print(f"Warning: {organ_name} mask missing for {img_full_path}")

            # Save merged mask
            merged_mask_output_path = mask_folder / "merged_masks.nii.gz"
            nib.save(nib.Nifti1Image(merged_mask_data, merged_mask_affine), merged_mask_output_path)

            all_files.append({'image': str(img_full_path), 'label': str(merged_mask_output_path)})

    print(f"Found {len(all_files)} valid image-mask pairs (skipped {skipped_count} of {total_count} total)")
    
    if len(all_files) == 0:
        raise ValueError("No valid image-mask pairs found. Please check your data.")

    train_files, val_files = train_test_split(
        all_files, test_size=val_split, random_state=seed
    )
    
    print(f"Training set: {len(train_files)} samples")
    print(f"Validation set: {len(val_files)} samples")

    train_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=['image', 'label'], 
            label_key='label', 
            spatial_size=(128, 128, 128),
            pos=1, 
            neg=1,
            num_samples=4
        ),
        RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.01),
        RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.8, 1.2)),
        RandAffined(
            keys=['image', 'label'], 
            prob=0.5, 
            rotate_range=(0.05, 0.05, 0.05), 
            scale_range=(0.05, 0.05, 0.05),
            mode=('bilinear', 'nearest')
        )
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, 
        batch_size=2, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=list_data_collate
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, train_files, val_files

def prepare_totalsegmentator_dataset(dataset_path, roi_subset,
                                     val_split=0.2, seed=42):
    """
    Prepare the data from TotalSegmentator dataset structure.
    
    Args:
        dataset_path (str or Path): Path to the dataset root
        roi_subset (list): List of organs to include in segmentation
        val_split (float): Validation split ratio
        seed (int): Random seed
        
    Returns:
        tuple: (train_loader, val_loader, train_files, val_files)
    """
    

    set_determinism(seed=seed)
    dataset_path = Path(dataset_path)
    
    # Find all subject folders
    subject_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('s')])
    print(f"Found {len(subject_dirs)} subject directories")
    
    all_files = []
    progress_bar = tqdm.tqdm(subject_dirs, desc="Processing subjects")
    
    for subject_dir in progress_bar:
        progress_bar.set_description(f"Processing {subject_dir.name}")
        
        # Check if CT file exists
        ct_file = subject_dir / "ct.nii.gz"
        if not ct_file.exists():
            progress_bar.write(f"CT file not found for {subject_dir}, skipping.")
            continue
            
        # Check if segmentation directory exists
        seg_dir = subject_dir / "segmentations"
        if not seg_dir.exists() or not seg_dir.is_dir():
            progress_bar.write(f"Segmentation directory not found for {subject_dir}, skipping.")
            continue
        
        # Check if merged mask already exists - skip processing
        merged_mask_path = seg_dir / "merged_kidneys.nii.gz"
        if merged_mask_path.exists():
            progress_bar.write(f"Merged mask already exists for {subject_dir.name}, using existing.")
            all_files.append({'image': str(ct_file), 'label': str(merged_mask_path)})
            continue
            
        # Check if all required segmentations exist
        missing_segmentations = False
        for organ in roi_subset:
            organ_file = seg_dir / f"{organ}.nii.gz"
            if not organ_file.exists():
                progress_bar.write(f"Segmentation for {organ} not found in {subject_dir}, skipping.")
                missing_segmentations = True
                break
                
        if missing_segmentations:
            continue
            
        # Create merged mask
        first_seg_path = seg_dir / f"{roi_subset[0]}.nii.gz"
        try:
            example_seg = nib.load(first_seg_path)
            merged_mask_data = np.zeros(example_seg.shape, dtype=np.uint8)
            merged_mask_affine = example_seg.affine
            
            # Merge all organ masks
            for idx, organ_name in enumerate(roi_subset, start=1):
                organ_path = seg_dir / f"{organ_name}.nii.gz"
                organ_mask = nib.load(organ_path).get_fdata()
                merged_mask_data[organ_mask > 0] = idx
                
            # Save merged mask
            nib.save(nib.Nifti1Image(merged_mask_data, merged_mask_affine), merged_mask_path)
            
            # Add to dataset
            all_files.append({'image': str(ct_file), 'label': str(merged_mask_path)})
        except Exception as e:
            progress_bar.write(f"Error processing {subject_dir.name}: {str(e)}")
    
    print(f"Found {len(all_files)} valid image-mask pairs")
    if len(all_files) == 0:
        raise ValueError("No valid image-mask pairs found. Please check your dataset path.")
        
    train_files, val_files = train_test_split(
        all_files, test_size=val_split, random_state=seed
    )
    
    print(f"Training set: {len(train_files)} samples")
    print(f"Validation set: {len(val_files)} samples")
    
    # Analyze foreground pixels
    print("Verifying dataset foreground pixels...")
    
    train_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(2.0, 2.0, 2.0), mode=('bilinear', 'nearest')),  # Lower resolution
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='label', margin=10),
        RandCropByPosNegLabeld(
            keys=['image', 'label'], 
            label_key='label', 
            spatial_size=(96, 96, 96),
            pos=2,
            neg=1,
            num_samples=2
        ),

        RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.01),
        RandAffined(
            keys=['image', 'label'], 
            prob=0.5, 
            rotate_range=(0.05, 0.05, 0.05), 
            scale_range=(0.05, 0.05, 0.05),
            mode=('bilinear', 'nearest'),
            padding_mode="zeros"
        )
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(2.0, 2.0, 2.0), mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='label', margin=10),  # Crop to reduce memory
    ])
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    print("Checking sample data for foreground pixels...")
    try:
        check_data = first(train_ds)
        label = check_data['label']
        unique_values = np.unique(label.numpy())
        print(f"Sample label contains classes: {unique_values}")
        foreground_count = np.sum(label.numpy() > 0)
        print(f"Foreground pixels in sample: {foreground_count}")
        if foreground_count == 0:
            print("WARNING: No foreground pixels found in sample! Check your masks.")
    except Exception as e:
        print(f"Error checking sample: {e}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=1,
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        collate_fn=list_data_collate
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_files, val_files