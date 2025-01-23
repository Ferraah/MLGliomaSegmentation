import os
import glob
import yaml
import logging
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.data import Dataset
import kagglehub

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ResizeD, ScaleIntensityd, 
    NormalizeIntensityd, RandRotate90d, RandFlipd, RandZoomd, RandAffined, 
    AsDiscreted, ToTensord, RandGaussianNoised
)

logger = logging.getLogger(__name__)

class GliomaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, channels=1):
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels
        
        # Download dataset
        logger.info("Downloading BraTS2020 dataset...")
        base_path = kagglehub.dataset_download(
            "darksteeldragon/brats2020-nifti-format-for-deepmedic"
        )
        
        # Set correct path to training data
        self.data_dir = os.path.join(
            base_path,
            "archive",
            "BraTS2020_TrainingData",
            "MICCAI_BraTS2020_TrainingData"
        )
        logger.info(f"Using data directory: {self.data_dir}")
        
        # Define transforms
        self.train_transforms = Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            ResizeD(keys=['image', 'label'], spatial_size=(192, 192, 192)),  # Resize the images
            ScaleIntensityd(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            RandRotate90d(keys=["image", "label"], prob=0.5),
            RandFlipd(keys=["image", "label"], prob=0.5),
            RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.8, max_zoom=1.2),
            RandAffined(keys=["image", "label"], prob=0.3),
            AsDiscreted(keys=["label"], to_onehot=5),
            ToTensord(keys=["image", "label"]),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)
        ])

        self.val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ResizeD(keys=['image', 'label'], spatial_size=(192, 192, 192)),  # Resize the images
            ScaleIntensityd(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            AsDiscreted(keys=["label"], to_onehot=5),
            ToTensord(keys=["image", "label"])
        ])

    def _find_subject_files(self):
        """Find all subject directories and their corresponding files."""
        subject_dirs = sorted(glob.glob(os.path.join(self.data_dir, "BraTS20_Training_*")))
        logger.info(f"Found {len(subject_dirs)} subject directories")
        
        data_pairs = []
        print("Channels: ", self.channels)

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            # Updated file paths to match BraTS2020 naming convention
            t1_file = os.path.join(subject_dir, f"{subject_id}_t1ce.nii")  # Changed from _t1 to _t1ce
            t2_file = os.path.join(subject_dir, f"{subject_id}_t2.nii")
            flair_file = os.path.join(subject_dir, f"{subject_id}_flair.nii")
            t1_native_file = os.path.join(subject_dir, f"{subject_id}_t1.nii")
            seg_file = os.path.join(subject_dir, f"{subject_id}_seg.nii")
            
            # Check if all required files exist
            if all(os.path.exists(f) for f in [t1_file, t2_file, flair_file, t1_native_file, seg_file]):
                
                if self.channels == 1:
                    data_pairs.append({
                        "image": t1_native_file,
                        "label": seg_file
                    })
                elif self.channels == 2:
                    data_pairs.append({
                        "image": [t1_native_file, t2_file],
                        "label": seg_file
                    })
                elif self.channels == 3:
                    data_pairs.append({
                        "image": [t1_native_file, t2_file, flair_file],
                        "label": seg_file
                    })
                else:
                    data_pairs.append({
                        # "image": [t1_file, t2_file, flair_file, t1_native_file],  # Include all modalities
                        "image": [t1_native_file, t2_file, flair_file, t1_file],
                        "label": seg_file
                    })
        
        if not data_pairs:
            raise ValueError("No valid data pairs found")
            
        logger.info(f"Found {len(data_pairs)} valid data pairs")
        return data_pairs

    def setup(self, stage=None):
        # Find all valid data pairs
        data_dicts = self._find_subject_files()
        
        # Split dataset
        train_val_dicts, test_dicts = train_test_split(
            data_dicts, test_size=0.2, random_state=42
        )
        train_dicts, val_dicts = train_test_split(
            train_val_dicts, test_size=0.2, random_state=42
        )
        
        logger.info(f"Train samples: {len(train_dicts)}")
        logger.info(f"Validation samples: {len(val_dicts)}")
        logger.info(f"Test samples: {len(test_dicts)}")
        
        # Save splits for reproducibility
        splits = {
            'train': [d['image'] for d in train_dicts],
            'validation': [d['image'] for d in val_dicts],
            'test': [d['image'] for d in test_dicts]
        }
        with open(f'splits-{self.channels}.yaml', 'w') as f:
            yaml.dump(splits, f)
        
        # Create MONAI datasets
        if stage == 'fit' or stage is None:
            self.train_ds = Dataset(train_dicts, transform=self.train_transforms)
            self.val_ds = Dataset(val_dicts, transform=self.val_transforms)
        
        if stage == 'test' or stage is None:
            self.test_ds = Dataset(test_dicts, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True
        )