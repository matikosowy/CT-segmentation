"""
Data Module (data.py)
-------------------
This module defines the dataset class and data loading functions for kidney segmentation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Kidneys2dDataset(Dataset):
    def __init__(
        self,
        patient_dirs,
        transforms=None,
        min_kidney_mask_pixels=50,
        split=None,
    ):
        """Dataset class for 2d slice-wise kidney segmentation.
        This class loads CT slices and their corresponding kidney masks, applying
        transformations if provided. It filters out slices where the kidney mask
        has fewer than `min_kidney_mask_pixels` pixels.

        Args:
            patient_dirs (Path): List of directories containing patient data.
            transforms: Transformations to apply to the images and masks.
            min_kidney_mask_pixels (int, optional): Minimum number of pixels in the kidney mask.
                Defaults to 50.
            split (str, optional): Split type ('train', 'val', 'test'). Defaults to None.
        """
        self.patient_dirs = patient_dirs
        self.transforms = transforms
        self.min_kidney_mask_pixels = min_kidney_mask_pixels
        self.split = split
        self.samples = []
        self._load_data()
        
    def _load_data(self):
        """Load and process CT slices meeting kidney mask criteria."""
        skipped_slices = 0
        
        for patient_dir in tqdm(self.patient_dirs, desc=f"Loading {self.split} patients"):
            image_path = patient_dir / "ct.nii.gz"
            left_mask_path = patient_dir / "segmentations/kidney_left.nii.gz"
            right_mask_path = patient_dir / "segmentations/kidney_right.nii.gz"
            
            image = nib.load(image_path).get_fdata()
            left_mask = nib.load(left_mask_path).get_fdata()
            right_mask = nib.load(right_mask_path).get_fdata()
             
            for slice_idx in range(image.shape[2]):
                image_slice = image[:, :, slice_idx]
                
                # HU window normalization
                image_slice = np.clip(image_slice, -100, 300)
                image_slice = (image_slice + 100) / 400
                
                left_mask_slice = left_mask[:, :, slice_idx]
                right_mask_slice = right_mask[:, :, slice_idx]
                
                combined_mask = np.zeros_like(left_mask_slice)
                combined_mask[left_mask_slice > 0] = 1
                combined_mask[right_mask_slice > 0] = 1
                
                if np.sum(combined_mask) >= self.min_kidney_mask_pixels:
                    self.samples.append({
                        'image': image_slice,
                        'mask': combined_mask,
                        'patient_id': patient_dir.name,
                        'slice_idx': slice_idx,
                    })
                else:
                    skipped_slices += 1
            
        print(f"Skipped {skipped_slices} slices due to small kidney masks.")
        print(f"{self.split.capitalize()}: Loaded {len(self.samples)} slices from {len(self.patient_dirs)} patients.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        mask = sample['mask']
        
        image = np.expand_dims(image, axis=-1).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1).astype(np.float32)
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # Restore original shape if transforms changed it
            if mask.ndim == 3 and mask.shape[-1] == 1:  # [H, W, C]
                mask = mask.permute(2, 0, 1)  # [C, H, W]
        
        return {
            'image': image,
            'mask': mask,
            'patient_id': sample['patient_id'],
        }


def create_kidney_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.15,
    test_ratio=0.15,
    num_patients=None,
    seed=42,
    min_kidney_mask_pixels=50,
    split="full",
    num_workers=4,
):
    """Creates DataLoader objects for training, validation, and testing of kidney segmentation.

    Args:
        root_dir (str): Path to the root directory containing patient data.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        val_ratio (float, optional): Ratio of validation set size to total dataset size. Defaults to 0.15.
        test_ratio (float, optional): Ratio of test set size to total dataset size. Defaults to 0.15.
        num_patients (_type_, optional): Number of patients to use. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        min_kidney_mask_pixels (int, optional): Minimum number of pixels in kidney mask. Defaults to 50.
        split (str, optional): Split type ('train', 'val', 'test', 'full'). Defaults to "full".
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.

    Returns:
        tuple: Tuple of DataLoader objects for train, val, and test splits.
    """
    root_dir = Path(root_dir)
    patient_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_dirs)
    
    if num_patients:
        patient_dirs = patient_dirs[:num_patients]
    
    # Split patients
    test_size = int(len(patient_dirs) * test_ratio)
    train_val_patients, test_patients = train_test_split(
        patient_dirs, test_size=test_size, random_state=seed
    )
    
    val_size = int(len(train_val_patients) * val_ratio / (1 - test_ratio))
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_size, random_state=seed
    )

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])
    
    common_loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    if split == "train":
        train_ds = Kidneys2dDataset(train_patients, train_transform, split='train', min_kidney_mask_pixels=min_kidney_mask_pixels)
        return DataLoader(train_ds, shuffle=True, **common_loader_args)
    elif split == "val":
        val_ds = Kidneys2dDataset(val_patients, val_transform, split='val', min_kidney_mask_pixels=min_kidney_mask_pixels)
        return DataLoader(val_ds, shuffle=False, **common_loader_args)
    elif split == "test":
        test_ds = Kidneys2dDataset(test_patients, val_transform, split='test', min_kidney_mask_pixels=min_kidney_mask_pixels)
        return DataLoader(test_ds, shuffle=False, **common_loader_args)
    else:
        train_ds = Kidneys2dDataset(train_patients, train_transform, split='train', min_kidney_mask_pixels=min_kidney_mask_pixels)
        val_ds = Kidneys2dDataset(val_patients, val_transform, split='val', min_kidney_mask_pixels=min_kidney_mask_pixels)
        test_ds = Kidneys2dDataset(test_patients, val_transform, split='test', min_kidney_mask_pixels=min_kidney_mask_pixels)

        return (
            DataLoader(train_ds, shuffle=True, **common_loader_args),
            DataLoader(val_ds, shuffle=False, **common_loader_args),
            DataLoader(test_ds, shuffle=False, **common_loader_args)
        )
