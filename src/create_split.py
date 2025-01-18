
from dataloader import get_data_path, get_data_loader

def create_split():
    data_path = get_data_path()
    data_loader = get_data_loader(data_path)
    
    return data_loader

