import os
import kagglehub
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeD,
    ToTensord,
    EnsureTyped,
    AsDiscreted,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
import json

set_determinism(seed=42)

class GliomaDataLoader:

    # Split the dataset into training, validation, and evaluation sets with custom percentages
    @staticmethod
    def split_dataset(dataset, train_pct, val_pct, save_path):
        train_size = int(train_pct * len(dataset))
        val_size = int(val_pct * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # Save the indices to files
        split_indices = {
            'train': train_dataset.indices,
            'val': val_dataset.indices,
            'test': test_dataset.indices
        }
        with open(os.path.join(save_path, 'split_indices.json'), 'w') as f:
            json.dump(split_indices, f)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def get_loaders(num_images=-1):
        path = kagglehub.dataset_download("darksteeldragon/brats2020-nifti-format-for-deepmedic")
        path = os.path.join(path, "archive", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")

        patients_folders = os.listdir(path)
        images = []
        labels = []
        for folder in patients_folders:
            images.append(os.path.join(path, folder, f"{folder}_t1.nii"))
            labels.append(os.path.join(path, folder, f"{folder}_seg.nii"))

        images = np.array(images)
        labels = np.array(labels)
        
        assert(len(images) > 0)
        assert(len(images) == len(labels))

        # Limit number of input images 
        if num_images != -1 and num_images < len(images):
            images = images[:num_images]
            labels = labels[:num_images]

        # Define the transforms
        transform = Compose([
            LoadImaged(keys=['image', 'label'], image_only=True),  # Load the images
            EnsureChannelFirstd(keys=['image', 'label']),  # Ensure the channels are first  
            ResizeD(keys=['image', 'label'], spatial_size=(128, 128, 128)),  # Resize the images
            ToTensord(keys=['image', 'label']),  # Convert the images to tensors
            EnsureTyped(keys=['image', 'label'], dtype=np.float32),  # Ensure the images are float32
            AsDiscreted(keys=['label'], to_onehot=5)  # Convert the labels to one-hot
        ])

        augments = Compose(
        [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])

        transform = Compose([transform, augments])
        # Create the dataset and data loader
        data_dicts = [{'image': image, 'label': label} for image, label in zip(images, labels)]
        dataset = Dataset(data=data_dicts, transform=transform)

        train_pct = 0.333
        val_pct = 0.333

        save_path = './'  # Change this to your desired save path
        os.makedirs(save_path, exist_ok=True)
        
        train_dataset, val_dataset, test_dataset = GliomaDataLoader.split_dataset(dataset, train_pct, val_pct, save_path)
        print("[DATA LOADER] Training set size:", len(train_dataset))
        print("[DATA LOADER] Validation set size:", len(val_dataset))
        print("[DATA LOADER] Test set size:", len(test_dataset))
        training_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
        validation_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
        return training_loader, validation_loader, test_loader
