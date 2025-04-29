import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings


class Kidneys2dDataset(Dataset):
    """
    2D Kidney CT slices Dataset with caching
    
    Args:
        root_dir (str or Path): Patients root directory
        transform (callable, optional): Transfomations to perform on the data
        combine_kidneys (bool): Whether to combine both kidney masks into one
        filter_empty (bool): Whether to skip empty kidney masks patients
        kidney_files (list): Kidney files names list, e.g. kidney_left.nii.gz
        window_width (int): CT HU window width
        window_center (int): CT HU window center
        min_kidney_pixels (int): Minimal kidney mask pixel count
        cache_dir (str or Path, optional): Cache dir path
        use_cache (bool): Whether to use caching
        val_ratio (float): Validation set percentage ratio
        test_ratio (float): Test set percentage ratio
        split (str): Which subset: train, val, test
        seed (int): Random seed
    """
    def __init__(self, 
                 root_dir, 
                 transform=None, 
                 combine_kidneys=True,
                 filter_empty=True,
                 kidney_files=None,
                 window_width=400,
                 window_center=50,
                 min_kidney_pixels=100,
                 cache_dir=None,
                 use_cache=True,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 split='train',
                 seed=42):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.combine_kidneys = combine_kidneys
        self.filter_empty = filter_empty
        self.kidney_files = kidney_files or ["kidney_right.nii.gz", "kidney_left.nii.gz"]
        self.window_width = window_width
        self.window_center = window_center
        self.min_kidney_pixels = min_kidney_pixels
        self.use_cache = use_cache
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        self.cache_file = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            if not cache_dir.exists():
                cache_dir.mkdir(exist_ok=True, parents=True)
            cache_filename = f"kidney_dataset_{'combined' if combine_kidneys else 'separate'}_cache.pt"
            self.cache_file = Path(cache_dir) / cache_filename
            
        self.samples = self._load_data()
        
        if split in ['train', 'val', 'test']:
            self._split_dataset()
            
    def _load_data(self):
        """2D slices NIFTI files loading"""
        if self.use_cache and self.cache_file and self.cache_file.exists():
            print(f"Loading from cache: {self.cache_file}")
            cached_data = torch.load(self.cache_file)
            return cached_data
            
        samples = []
        patient_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        print(f"Found {len(patient_dirs)} patient directories...")
        for patient_dir in tqdm(patient_dirs, desc="Patients processing"):
            patient_path = Path(patient_dir)
            ct_path = patient_path / "ct.nii.gz"
            
            if not ct_path.exists():
                continue
                
            mask_paths = []
            if patient_path.name.startswith('s'):  #TotalSegmentator dataset format
                seg_dir = patient_path / "segmentations"
                if not seg_dir.exists():
                    continue
                mask_paths = [seg_dir / mask_file for mask_file in self.kidney_files]
            else:
                mask_paths = [patient_path / mask_file for mask_file in self.kidney_files]
                
            if not all(mask_path.exists() for mask_path in mask_paths):
                continue
                
            try:
                ct_nifti = nib.load(str(ct_path))
                ct_volume = ct_nifti.get_fdata()
                
                min_value = self.window_center - self.window_width//2
                max_value = self.window_center + self.window_width//2
                ct_volume = np.clip(ct_volume, min_value, max_value)
                ct_volume = (ct_volume - min_value) / (max_value - min_value)
                
                mask_volumes = []
                mask_volume_shapes = []
                for mask_path in mask_paths:
                    mask_nifti = nib.load(str(mask_path))
                    mask_volume = mask_nifti.get_fdata()
                    mask_volumes.append(mask_volume)
                    mask_volume_shapes.append(mask_volume.shape)
                
                if len(set(map(tuple, mask_volume_shapes))) > 1 or ct_volume.shape != mask_volumes[0].shape:
                    print(f"Skipped {patient_dir} - uneven shapes, ct: {ct_volume.shape}, mask: {mask_volume_shapes}")
                    continue
                
                depth = ct_volume.shape[2]
                for z_idx in range(depth):
                    ct_slice = ct_volume[:, :, z_idx]
                    
                    organ_masks = []
                    for mask_volume in mask_volumes:
                        organ_masks.append(mask_volume[:, :, z_idx])
                    
                    if self.filter_empty:
                        total_kidney_pixels = sum(np.sum(mask > 0) for mask in organ_masks)
                        if total_kidney_pixels < self.min_kidney_pixels:
                            continue
                    
                    slice_data = {
                        'patient_id': patient_dir.name,
                        'slice_idx': z_idx,
                        'ct_slice': ct_slice,
                        'organ_masks': organ_masks
                    }
                    
                    samples.append(slice_data)
                
            except Exception as e:
                print(f"Processing error {patient_dir}: {str(e)}")
        
        print(f"Found {len(samples)} kidney slices")
        
        if self.cache_file is not None:
            print(f"Saving to cache: {self.cache_file}")
            torch.save(samples, self.cache_file)
            
        return samples
        
    def _split_dataset(self):
        """Training, validation and test sets split (patient-wise split)"""
        patient_ids = list(set(sample['patient_id'] for sample in self.samples))

        train_val_patients, test_patients = train_test_split(
            patient_ids, test_size=self.test_ratio, random_state=self.seed
        )
        
        if self.val_ratio > 0:
            val_ratio_adjusted = self.val_ratio / (1 - self.test_ratio)
            train_patients, val_patients = train_test_split(
                train_val_patients, test_size=val_ratio_adjusted, random_state=self.seed
            )
        else:
            train_patients, val_patients = train_val_patients, []
            
        if self.split == 'train':
            self.samples = [s for s in self.samples if s['patient_id'] in train_patients]
        elif self.split == 'val':
            self.samples = [s for s in self.samples if s['patient_id'] in val_patients]
        elif self.split == 'test':
            self.samples = [s for s in self.samples if s['patient_id'] in test_patients]
            
        print(f"Subset: '{self.split}': {len(self.samples)} slices")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = sample['ct_slice'].astype(np.float32)
        organ_masks = sample['organ_masks']
        
        if self.combine_kidneys:
            mask = np.zeros_like(image, dtype=np.float32)
            for organ_mask in organ_masks:
                mask = np.logical_or(mask, organ_mask > 0).astype(np.float32)
            mask = mask[np.newaxis, :, :] # channel dim add
        else:
            masks = [organ_mask.astype(np.float32) for organ_mask in organ_masks]
            mask = np.stack(masks, axis=0)
        
        image = image[np.newaxis, :, :]
            
        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0), 
                                        mask=mask.transpose(1, 2, 0))
            image = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask'].transpose(2, 0, 1)
        
        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'patient_id': sample['patient_id'],
            'slice_idx': sample['slice_idx']
        }


def create_kidney_dataloaders(root_dir,
                             batch_size=16,
                             num_workers=4,
                             combine_kidneys=True,
                             cache_dir=None,
                             val_ratio=0.15,
                             test_ratio=0.15,
                             seed=42):
    """
    Creates segmentation dataloaders for: train, val, test sets
    
    Args:
        root_dir (str): Data directory path
        batch_size (int): Number of samples in a batch
        num_workers (int): Processing units count
        combine_kidneys (bool): Whether to combine both kidney masks into one
        cache_dir (str): Cache dir path
        val_ratio (float): Validation set percentage ratio
        test_ratio (float): Test set percentage ratio
        seed (int): Random seed
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    train_transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,), p=1.0),
        A.OneOf([
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=60, sigma=60 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, p=0.5),
        ], p=0.3),
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.Resize(512, 512, p=1.0),
        A.GaussNoise(p=0.2, std_range=(0.01, 0.05)),
    ])
    
    # normalization only
    val_transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,), p=1.0),
        A.Resize(512, 512, p=1.0),
    ])
    
    train_ds = Kidneys2dDataset(
        root_dir=root_dir,
        transform=train_transform,
        combine_kidneys=combine_kidneys,
        filter_empty=True,
        cache_dir=cache_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split='train',
        seed=seed
    )
    
    val_ds = Kidneys2dDataset(
        root_dir=root_dir,
        transform=val_transform,
        combine_kidneys=combine_kidneys,
        filter_empty=True,
        cache_dir=cache_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split='val',
        seed=seed
    )
    
    test_ds = Kidneys2dDataset(
        root_dir=root_dir,
        transform=val_transform,
        combine_kidneys=combine_kidneys,
        filter_empty=True,
        cache_dir=cache_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split='test',
        seed=seed
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader