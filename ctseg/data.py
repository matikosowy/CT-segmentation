"""
Data Module (data.py)
-------------------
This module defines the dataset class and data loading functions for kidney segmentation.
"""

from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import albumentations as A
import monai.transforms as transforms_3d
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CT2dDataset(Dataset):
    def __init__(
        self,
        patient_dirs,
        target_organs_names=["kidney_left", "kidney_right"],
        min_organs_mask_pixels=[50, 50],
        transforms=None,
        split=None,
        reset_cache=False,
    ):
        """
        Dataset class for 2d slice-wise medical segmentation.
        This class loads CT slices and their corresponding organ masks, applying
        transformations if provided. It filters out slices where the organ's masks
        has fewer than `min_organs_mask_pixels` pixels.

        Args:
            patient_dirs (Path): List of directories containing patient data.
            target_organs_names (list): List of organ names to segment.
            min_organs_mask_pixels (list): Minimum number of pixels in organ masks.
            transforms: Transformations to apply to the images and masks.
            split (str, optional): Split type ('train', 'val', 'test'). Defaults to None.
            reset_cache (bool, optional): If True, reset the cache. Defaults to False.
        """
        self.patient_dirs = patient_dirs
        self.target_organs = target_organs_names
        self.min_target_masks = min_organs_mask_pixels
        self.transforms = transforms
        self.split = split
        self.samples = []

        if len(self.target_organs) != len(self.min_target_masks):
            raise ValueError(
                f"""target_organs and min_organ_pixels must have same length.
                Got {len(self.target_organs)} and {len(self.min_target_masks)}"""
            )

        num_patients = len(self.patient_dirs)
        organs_str = "_".join(self.target_organs)
        cache_path = Path(f"cache/cache_{self.split}_{num_patients}_{organs_str}.pt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not reset_cache:
            print(f"Loading cached {split} data from '{cache_path}'...")
            self.samples = torch.load(cache_path)
            print(f"Loaded {len(self.samples)} slices from cache.")
            return
        else:
            if reset_cache:
                print(f"Resetting cache. Deleting '{cache_path}'...")
                cache_path.unlink(missing_ok=True)
                self._load_data()
                self._cache_data(cache_path)
            else:
                print(f"Cache for {split} data not found. Loading data...")
                self._load_data()
                self._cache_data(cache_path)

    def _load_data(self):
        """Load and process CT slices meeting organ mask criteria."""
        skipped_slices = 0

        print("=" * 50)

        for patient_dir in tqdm(self.patient_dirs, desc=f"Loading {self.split} patients"):
            image_path = patient_dir / "ct.nii.gz"
            image = nib.load(image_path).get_fdata()

            organ_masks = []

            for organ_name in self.target_organs:
                mask_path = patient_dir / "segmentations" / f"{organ_name}.nii.gz"

                if mask_path.exists():
                    mask = nib.load(mask_path).get_fdata()
                    organ_masks.append(mask)
                else:
                    print(f"Warning: Mask for {organ_name} not found in {patient_dir}!")
                    organ_masks.append(np.zeros_like(image))

            for slice_idx in range(image.shape[2]):
                image_slice = image[:, :, slice_idx]

                # HU window normalization
                image_slice = np.clip(image_slice, -100, 300)
                image_slice = (image_slice + 100) / 400

                mask_slices = []
                include_slice = False

                for i, (organ_mask, min_pixels) in enumerate(zip(organ_masks, self.min_target_masks)):
                    mask_slice = organ_mask[:, :, slice_idx]
                    mask_slice = np.clip(mask_slice, 0, 1)
                    mask_slices.append(mask_slice)

                    if np.sum(mask_slice) >= min_pixels:
                        include_slice = True

                # Only include slice if any organ mask meets the pixel criteria
                if include_slice:
                    combined_masks = np.stack(mask_slices, axis=-1).astype(np.float32)

                    self.samples.append(
                        {
                            "image": image_slice,
                            "mask": combined_masks,
                            "patient_id": patient_dir.name,
                            "slice_idx": slice_idx,
                        }
                    )
                else:
                    skipped_slices += 1

        print(f"Skipped {skipped_slices} slices due to insufficient masks.")
        print(f"{self.split.capitalize()}: Loaded {len(self.samples)} slices from {len(self.patient_dirs)} patients.")

    def _cache_data(self, cache_path):
        """Cache the loaded data to a file."""
        torch.save(self.samples, cache_path)
        print(f"Data cached to '{cache_path}'.\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        masks = sample["mask"]

        image = np.expand_dims(image, axis=-1).astype(np.float32)  # [H, W, 1]

        if self.transforms:
            augmented = self.transforms(image=image, mask=masks)
            image = augmented["image"]
            masks = augmented["mask"]

            if masks.ndim == 3 and masks.shape[-1] > 1:
                masks = masks.permute(2, 0, 1)  # [C, H, W]

        return {
            "image": image,
            "mask": masks,
            "patient_id": sample["patient_id"],
            "slice_idx": sample["slice_idx"],
        }


def create_2d_segmentation_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.15,
    test_ratio=0.15,
    num_patients=None,
    seed=42,
    split="full",
    num_workers=4,
    target_organs=["kidney_left", "kidney_right"],
    min_organ_pixels=[50, 50],
    reset_cache=False,
):
    """Creates DataLoader objects for training, validation, and testing of kidney segmentation.

    Args:
        root_dir (str): Path to the root directory containing patient data.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        val_ratio (float, optional): Ratio of validation set size to total dataset size. Defaults to 0.15.
        test_ratio (float, optional): Ratio of test set size to total dataset size. Defaults to 0.15.
        num_patients (_type_, optional): Number of patients to use. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        split (str, optional): Split type ('train', 'val', 'test', 'full'). Defaults to "full".
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
        target_organs (list, optional): List of organs to segment. Defaults to ["kidney_left", "kidney_right"].
        min_organ_pixels (list, optional): Minimum number of pixels for each organ mask. Defaults to [50, 50].
        reset_cache (bool, optional): If True, reset the dataset file cache. Defaults to False.

    Returns:
        tuple: Tuple of DataLoader objects for train, val, and test splits.
    """
    print("=" * 50)
    print("DATA PROCESSING")

    root_dir = Path(root_dir)
    patient_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_dirs)

    # Split patients
    if split == "train":
        train_patients = patient_dirs[:num_patients]
    elif split == "val":
        patient_dirs = patient_dirs[200:]
        val_patients = patient_dirs[:num_patients]
    elif split == "test":
        patient_dirs = patient_dirs[300:]
        test_patients = patient_dirs[:num_patients]
    else:
        patient_dirs = patient_dirs[:num_patients]

        test_size = int(len(patient_dirs) * test_ratio)
        train_val_patients, test_patients = train_test_split(patient_dirs, test_size=test_size, random_state=seed)

        val_size = int(len(train_val_patients) * val_ratio / (1 - test_ratio))
        val_size = min(val_size, 10)  # Limit to 10 patients for validation (speed up training)

        train_patients, val_patients = train_test_split(train_val_patients, test_size=val_size, random_state=seed)

    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                p=0.3,
            ),
            A.GaussNoise(p=0.2, std_range=(0.01, 0.1)),
            A.GridDistortion(p=0.2, distort_limit=(-0.1, 0.1)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(256, 256),
            ToTensorV2(),
        ]
    )

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    if split == "train":
        train_ds = CT2dDataset(
            train_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=train_transform,
            split="train",
            reset_cache=reset_cache,
        )
        return DataLoader(train_ds, shuffle=True, **common_loader_args)
    elif split == "val":
        val_ds = CT2dDataset(
            val_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="val",
            reset_cache=reset_cache,
        )
        return DataLoader(val_ds, shuffle=False, **common_loader_args)
    elif split == "test":
        test_ds = CT2dDataset(
            test_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="test",
            reset_cache=reset_cache,
        )
        return DataLoader(test_ds, shuffle=False, **common_loader_args)
    else:
        train_ds = CT2dDataset(
            train_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=train_transform,
            split="train",
            reset_cache=reset_cache,
        )
        val_ds = CT2dDataset(
            val_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="val",
            reset_cache=reset_cache,
        )
        test_ds = CT2dDataset(
            test_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="test",
            reset_cache=reset_cache,
        )

        return (
            DataLoader(train_ds, shuffle=True, **common_loader_args),
            DataLoader(val_ds, shuffle=False, **common_loader_args),
            DataLoader(test_ds, shuffle=False, **common_loader_args),
        )


class CT3dDataset(Dataset):
    def __init__(
        self,
        patient_dirs,
        target_organs_names=["kidney_left", "kidney_right"],
        min_organs_mask_pixels=[200, 200],
        height=8,
        transforms=None,
        split=None,
        reset_cache=False,
    ):
        """
        Dataset class for 3D kidney segmentation.
        This class loads 3D CT volume fragments and their corresponding organ masks.

        Args:
            patient_dirs (Path): List of directories containing patient data.
            target_organs_names (list): List of organ names to segment.
            height (int): Height of the 3D volume fragment.
            min_organs_mask_pixels (list): Minimum number of pixels in organ masks.
            transforms: Transformations to apply to the images and masks.
            split (str, optional): Split type ('train', 'val', 'test'). Defaults to None.
            reset_cache (bool, optional): If True, reset the cache. Defaults to False.
        """
        self.patient_dirs = patient_dirs
        self.target_organs = target_organs_names
        self.height = height
        self.min_target_masks = min_organs_mask_pixels
        self.transforms = transforms
        self.split = split
        self.samples = []

        if len(self.target_organs) != len(self.min_target_masks):
            raise ValueError(
                f"""target_organs and min_organ_pixels must have same length.
                Got {len(self.target_organs)} and {len(self.min_target_masks)}"""
            )

        num_patients = len(self.patient_dirs)
        organs_str = "_".join(self.target_organs)
        cache_path = Path(f"cache/cache3d_{self.split}_{num_patients}_{organs_str}.pt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not reset_cache:
            print(f"Loading cached {split} data from '{cache_path}'...")
            self.samples = torch.load(cache_path)
            print(f"Loaded {len(self.samples)} slices from cache.")
            return
        else:
            if reset_cache:
                print(f"Resetting cache. Deleting '{cache_path}'...")
                cache_path.unlink(missing_ok=True)
                self._load_data()
                self._cache_data(cache_path)
            else:
                print(f"Cache for {split} data not found. Loading data...")
                self._load_data()
                self._cache_data(cache_path)

    def _load_data(self):
        """Load and process 3D CT volume fragments meeting organ mask criteria."""
        skipped_volumes = 0
        volume_idx = -1

        print("=" * 50)

        for patient_dir in tqdm(self.patient_dirs, desc=f"Loading {self.split} patients"):
            image_path = patient_dir / "ct.nii.gz"
            image = nib.load(image_path).get_fdata()

            organ_masks = []

            for organ_name in self.target_organs:
                mask_path = patient_dir / "segmentations" / f"{organ_name}.nii.gz"

                if mask_path.exists():
                    mask = nib.load(mask_path).get_fdata()
                    organ_masks.append(mask)
                else:
                    print(f"Warning: Mask for {organ_name} not found in {patient_dir}!")
                    organ_masks.append(np.zeros_like(image))

            step = max(self.height // 2, 1)  # overlap volumes
            for slice_idx in range(0, image.shape[2] - self.height + 1, step):
                volume_fragment = image[:, :, slice_idx : slice_idx + self.height]

                # Last fragment may be smaller than height
                if volume_fragment.shape[2] < self.height:
                    continue

                mask_fragments = []
                include_fragment = False

                # Check if the fragment has enough pixels
                for i, (organ_mask, min_pixels) in enumerate(zip(organ_masks, self.min_target_masks)):
                    mask_fragment = organ_mask[:, :, slice_idx : slice_idx + self.height]
                    mask_fragment = np.clip(mask_fragment, 0, 1)
                    mask_fragments.append(mask_fragment.astype(np.float32))

                    if np.sum(mask_fragment) >= min_pixels:
                        include_fragment = True

                if not include_fragment:
                    skipped_volumes += 1
                    continue

                # If mask is enough, normalize the image fragment
                volume_fragment = np.clip(volume_fragment, -100, 300)
                volume_fragment = (volume_fragment + 100) / 400

                combined_masks = np.stack(mask_fragments, axis=-1)  # [H, W, D, C]

                volume_idx += 1

                self.samples.append(
                    {
                        "image": volume_fragment,
                        "mask": combined_masks,
                        "patient_id": patient_dir.name,
                        "start_slice": slice_idx,
                        "end_slice": slice_idx + self.height - 1,
                        "volume_idx": volume_idx,
                    }
                )

        print(f"Skipped {skipped_volumes} volumes due to insufficient masks.")
        print(f"{self.split.capitalize()}: Loaded {len(self.samples)} slices from {len(self.patient_dirs)} patients.")

    def _cache_data(self, cache_path):
        """Cache the loaded data to a file."""
        torch.save(self.samples, cache_path)
        print(f"Data cached to '{cache_path}'.\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        masks = sample["mask"]

        image = np.expand_dims(image, axis=-1).astype(np.float32)  # [H, W, D, 1]

        if self.transforms:
            # Monai transforms for 3d
            augmented = self.transforms(image=image, mask=masks)
            image = augmented["image"]
            masks = augmented["mask"]

            if masks.ndim == 4 and masks.shape[-1] > 1:
                masks = masks.permute(3, 0, 1, 2)  # [C, H, W, D]

        return {
            "image": image,
            "mask": masks,
            "patient_id": sample["patient_id"],
            "volume_idx": sample["volume_idx"],
        }


def create_3d_segmentation_dataloaders(
    root_dir,
    batch_size=4,
    val_ratio=0.15,
    test_ratio=0.15,
    num_patients=None,
    seed=42,
    split="full",
    num_workers=4,
    target_organs=["kidney_left", "kidney_right"],
    min_organ_pixels=[200, 200],
    reset_cache=False,
    height=8,
):
    """Creates DataLoader objects for training, validation, and testing of kidney segmentation.

    Args:
        root_dir (str): Path to the root directory containing patient data.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 4.
        val_ratio (float, optional): Ratio of validation set size to total dataset size. Defaults to 0.15.
        test_ratio (float, optional): Ratio of test set size to total dataset size. Defaults to 0.15.
        num_patients (_type_, optional): Number of patients to use. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        split (str, optional): Split type ('train', 'val', 'test', 'full'). Defaults to "full".
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
        target_organs (list, optional): List of organs to segment. Defaults to ["kidney_left", "kidney_right"].
        min_organ_pixels (list, optional): Minimum number of pixels for each organ mask. Defaults to [200, 200]].
        reset_cache (bool, optional): If True, reset the dataset file cache. Defaults to False.
        height (int, optional): Height of the 3D volume fragment. Defaults to 8.

    Returns:
        tuple: Tuple of DataLoader objects for train, val, and test splits.
    """
    print("=" * 50)
    print("DATA PROCESSING")

    root_dir = Path(root_dir)
    patient_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_dirs)

    # Split patients
    if split == "train":
        train_patients = patient_dirs[:num_patients]
    elif split == "val":
        patient_dirs = patient_dirs[200:]
        val_patients = patient_dirs[:num_patients]
    elif split == "test":
        patient_dirs = patient_dirs[300:]
        test_patients = patient_dirs[:num_patients]
    else:
        patient_dirs = patient_dirs[:num_patients]

        test_size = int(len(patient_dirs) * test_ratio)
        train_val_patients, test_patients = train_test_split(patient_dirs, test_size=test_size, random_state=seed)

        val_size = int(len(train_val_patients) * val_ratio / (1 - test_ratio))
        val_size = min(val_size, 10)  # Limit to 10 patients for validation (speed up training)

        train_patients, val_patients = train_test_split(train_val_patients, test_size=val_size, random_state=seed)

    train_transform = transforms_3d.Compose(
        [
            transforms_3d.Resize((height, 256, 256)),
            transforms_3d.RandAffined(
                prob=0.3,
                rotate_range=(-15, 15),
                scale_range=(0.8, 1.2),
                translate_range=(-0.1, 0.1),
            ),
            transforms_3d.GaussianNoise(prob=0.2, std=(0.01, 0.1)),
            transforms_3d.GridDistortion(prob=0.2, distort_limit=(-0.1, 0.1)),
            transforms_3d.ToTensor(),
        ]
    )

    val_transform = transforms_3d.Compose(
        [
            transforms_3d.Resize((height, 256, 256)),
            transforms_3d.ToTensor(),
        ]
    )

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    if split == "train":
        train_ds = CT3dDataset(
            train_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=train_transform,
            split="train",
            reset_cache=reset_cache,
            height=height,
        )
        return DataLoader(train_ds, shuffle=True, **common_loader_args)
    elif split == "val":
        val_ds = CT3dDataset(
            val_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="val",
            reset_cache=reset_cache,
            height=height,
        )
        return DataLoader(val_ds, shuffle=False, **common_loader_args)
    elif split == "test":
        test_ds = CT3dDataset(
            test_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="test",
            reset_cache=reset_cache,
            height=height,
        )
        return DataLoader(test_ds, shuffle=False, **common_loader_args)
    else:
        train_ds = CT3dDataset(
            train_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=train_transform,
            split="train",
            reset_cache=reset_cache,
            height=height,
        )
        val_ds = CT3dDataset(
            val_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="val",
            reset_cache=reset_cache,
            height=height,
        )
        test_ds = CT3dDataset(
            test_patients,
            target_organs_names=target_organs,
            min_organs_mask_pixels=min_organ_pixels,
            transforms=val_transform,
            split="test",
            reset_cache=reset_cache,
            height=height,
        )

        return (
            DataLoader(train_ds, shuffle=True, **common_loader_args),
            DataLoader(val_ds, shuffle=False, **common_loader_args),
            DataLoader(test_ds, shuffle=False, **common_loader_args),
        )
