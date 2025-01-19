import os
import kagglehub
import nibabel as nib
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeD,
    ToTensord,
    EnsureTyped,
    AsDiscreted
)

set_determinism(seed=42)

class GliomaDataLoader:

    @staticmethod
    def get_training_loader():
        path = kagglehub.dataset_download("darksteeldragon/brats2020-nifti-format-for-deepmedic")
        path = os.path.join(path, "archive", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
        return GliomaDataLoader.get_loader(path)

    @staticmethod
    def get_validation_loader():
        path = kagglehub.dataset_download("darksteeldragon/brats2020-nifti-format-for-deepmedic")
        path = os.path.join(path, "archive", "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData")
        return GliomaDataLoader.get_loader(path)

    @staticmethod
    def get_loader(path):
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

        images = images[:2]
        labels = labels[:2]
        # Define the transforms

        transform = Compose([
            LoadImaged(keys=['image', 'label'], image_only=True),  # Load the images
            EnsureChannelFirstd(keys=['image', 'label']),  # Ensure the channels are first  
            ResizeD(keys=['image', 'label'], spatial_size=(128, 128, 128)),  # Resize the images
            ToTensord(keys=['image', 'label']),  # Convert the images to tensors
            EnsureTyped(keys=['image', 'label'], dtype=np.float32),  # Ensure the images are float32
            AsDiscreted(keys=['label'], to_onehot=5)  # Convert the labels to one-hot
        ])

        # Create the dataset and data loader
        data_dicts = [{'image': image, 'label': label} for image, label in zip(images, labels)]
        dataset = Dataset(data=data_dicts, transform=transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
        return loader
