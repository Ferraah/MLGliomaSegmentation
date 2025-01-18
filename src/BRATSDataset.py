
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity, ToTensor

class BRATSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx])
        mask = sitk.ReadImage(self.mask_paths[idx])
        
        if self.transform:
            # Apply transformations (e.g., resizing, normalization)
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
