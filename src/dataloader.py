from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
import kagglehub

def get_data_path():
    # Download latest version
    path = kagglehub.dataset_download("darksteeldragon/brats2020-nifti-format-for-deepmedic")
    print("Path to dataset files:", path)
    return path

def get_data_loader(data_path):
    # Create dataset
    dataset = Dataset(data=data_path, transform=None)
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    return data_loader
