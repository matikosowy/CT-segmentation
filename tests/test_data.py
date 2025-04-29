import unittest
import tempfile
import shutil
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ctseg.data import Kidneys2dDataset, create_kidney_dataloaders

class TestKidneys2dDataset(unittest.TestCase):
    """Tests for Kidneys2dDataset class"""
    def setUp(self):
        """Test data prep"""
        self.test_dir = tempfile.mkdtemp()
        
        self.patients = ["patient1", "patient2", "s1234"]  # sXXXX for TotalSegmentator
        
        for patient in self.patients:
            patient_dir = Path(self.test_dir) / patient
            os.makedirs(patient_dir, exist_ok=True)
            
            ct_shape = (64, 64, 16)
            ct_data = np.random.rand(*ct_shape) * 1000 - 500
            ct_img = nib.Nifti1Image(ct_data, np.eye(4))
            nib.save(ct_img, str(patient_dir / "ct.nii.gz"))
            
            if patient.startswith('s'):
                seg_dir = patient_dir / "segmentations"
                os.makedirs(seg_dir, exist_ok=True)
                
                kidney_right = np.zeros(ct_shape, dtype=np.uint8)
                kidney_right[20:30, 20:30, 6:10] = 1
                
                kidney_left = np.zeros(ct_shape, dtype=np.uint8)
                kidney_left[20:30, 35:45, 6:10] = 1 
                
                kidney_right_img = nib.Nifti1Image(kidney_right, np.eye(4))
                kidney_left_img = nib.Nifti1Image(kidney_left, np.eye(4))
                
                nib.save(kidney_right_img, str(seg_dir / "kidney_right.nii.gz"))
                nib.save(kidney_left_img, str(seg_dir / "kidney_left.nii.gz"))
            else:
                kidney_right = np.zeros(ct_shape, dtype=np.uint8)
                kidney_right[20:30, 20:30, 6:10] = 1
                
                kidney_left = np.zeros(ct_shape, dtype=np.uint8)
                kidney_left[20:30, 35:45, 6:10] = 1
                
                kidney_right_img = nib.Nifti1Image(kidney_right, np.eye(4))
                kidney_left_img = nib.Nifti1Image(kidney_left, np.eye(4))
                
                nib.save(kidney_right_img, str(patient_dir / "kidney_right.nii.gz"))
                nib.save(kidney_left_img, str(patient_dir / "kidney_left.nii.gz"))
        
        self.cache_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup after tests"""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.cache_dir)
    
    def test_dataset_creation(self):
        dataset = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=True,
            filter_empty=True,
            cache_dir=None,
            use_cache=False
        )
        
        self.assertGreater(len(dataset), 0, "Dataset is empty!")
        
        train_dataset = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=True,
            filter_empty=True,
            cache_dir=None,
            use_cache=False,
            split='train'
        )
        
        val_dataset = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=True,
            filter_empty=True,
            cache_dir=None,
            use_cache=False,
            split='val'
        )
        
        total_slices = len(train_dataset) + len(val_dataset)
        self.assertGreater(total_slices, 0, "No slices found for train and val datasets")
    
    def test_getitem(self):
        dataset = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=True,
            filter_empty=True,
            cache_dir=None,
            use_cache=False
        )
        
        sample = dataset[0]
        
        self.assertIn('image', sample, "No 'image' key in returned sample")
        self.assertIn('mask', sample, "No 'mask' key in returned sample")
        self.assertIn('patient_id', sample, "No 'patient_id' key in returned sample")
        self.assertIn('slice_idx', sample, "No 'slice_idx' key in returned sample")
        
        self.assertEqual(sample['image'].dim(), 3, "Image should have 3 dimensions (channel, height, width)")
        self.assertEqual(sample['mask'].dim(), 3, "Mask should have 3 dimensions (channel, height, width)")
        
        self.assertTrue(isinstance(sample['image'], torch.Tensor), "Image should be a PyTorch tensor")
        self.assertTrue(isinstance(sample['mask'], torch.Tensor), "Mask should be a PyTorch tensor")
    
    def test_cache(self):
        dataset1 = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=True,
            filter_empty=True,
            cache_dir=self.cache_dir,
            use_cache=True
        )
        
        cache_files = list(Path(self.cache_dir).glob('*.pt'))
        self.assertEqual(len(cache_files), 1, "Cache file not created")
        
        dataset2 = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=True,
            filter_empty=True,
            cache_dir=self.cache_dir,
            use_cache=True
        )
        
        self.assertEqual(len(dataset1), len(dataset2), "Datasets are not equal")
    
    def test_combine_kidneys(self):
        dataset_combined = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=True,
            filter_empty=True,
            cache_dir=None,
            use_cache=False
        )
        
        dataset_separate = Kidneys2dDataset(
            root_dir=self.test_dir,
            combine_kidneys=False,
            filter_empty=True,
            cache_dir=None,
            use_cache=False
        )
        
        sample_combined = dataset_combined[0]
        sample_separate = dataset_separate[0]
        
        self.assertEqual(sample_combined['mask'].shape[0], 1, 
                         "Combined kidneys should have one channel")
        self.assertEqual(sample_separate['mask'].shape[0], 2, 
                         "Separate kidneys should have two channels")


class TestDataLoaders(unittest.TestCase):

    def setUp(self):
        """Test data preparation for DataLoader tests"""
        self.test_dir = tempfile.mkdtemp()
        
        patients = ["patient1", "patient2", "s1234"]
        
        for patient in patients:
            patient_dir = Path(self.test_dir) / patient
            os.makedirs(patient_dir, exist_ok=True)
            
            ct_shape = (64, 64, 16)
            ct_data = np.random.rand(*ct_shape) * 1000 - 500
            ct_img = nib.Nifti1Image(ct_data, np.eye(4))
            nib.save(ct_img, str(patient_dir / "ct.nii.gz"))
            
            if patient.startswith('s'):
                seg_dir = patient_dir / "segmentations"
                os.makedirs(seg_dir, exist_ok=True)
                
                kidney_right = np.zeros(ct_shape, dtype=np.uint8)
                kidney_right[20:30, 20:30, 6:10] = 1
                
                kidney_left = np.zeros(ct_shape, dtype=np.uint8)
                kidney_left[20:30, 35:45, 6:10] = 1
                
                kidney_right_img = nib.Nifti1Image(kidney_right, np.eye(4))
                kidney_left_img = nib.Nifti1Image(kidney_left, np.eye(4))
                
                nib.save(kidney_right_img, str(seg_dir / "kidney_right.nii.gz"))
                nib.save(kidney_left_img, str(seg_dir / "kidney_left.nii.gz"))
            else:
                kidney_right = np.zeros(ct_shape, dtype=np.uint8)
                kidney_right[20:30, 20:30, 6:10] = 1
                
                kidney_left = np.zeros(ct_shape, dtype=np.uint8)
                kidney_left[20:30, 35:45, 6:10] = 1
                
                kidney_right_img = nib.Nifti1Image(kidney_right, np.eye(4))
                kidney_left_img = nib.Nifti1Image(kidney_left, np.eye(4))
                
                nib.save(kidney_right_img, str(patient_dir / "kidney_right.nii.gz"))
                nib.save(kidney_left_img, str(patient_dir / "kidney_left.nii.gz"))
    
    def tearDown(self):
        """Cleanup after tests"""
        shutil.rmtree(self.test_dir)
    
    def test_create_dataloaders(self):
        train_loader, val_loader, test_loader = create_kidney_dataloaders(
            root_dir=self.test_dir,
            batch_size=4,
            num_workers=0,
            combine_kidneys=True,
            cache_dir=None
        )
        
        self.assertIsInstance(train_loader, DataLoader, "train_loader is not an instance of DataLoader")
        self.assertIsInstance(val_loader, DataLoader, "val_loader is not an instance of DataLoader")
        self.assertIsInstance(test_loader, DataLoader, "test_loader is not an instance of DataLoader")
        
        self.assertEqual(train_loader.batch_size, 4, "Invalid batch size for train_loader")
        
        try:
            batch = next(iter(train_loader))
            self.assertIn('image', batch, "No 'image' key in returned batch")
            self.assertIn('mask', batch, "No 'mask' key in returned batch")
        except Exception as e:
            self.fail(f"Exception during iteration through train_loader: {str(e)}")


if __name__ == "__main__":
    unittest.main()