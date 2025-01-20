

from test import test
from train import train
import os

if __name__ == "__main__":
    if os.path.exists("split_indices.json"):
        # remove
        os.remove("split_indices.json")

    num_images = 100
    train(max_epochs=20, num_images=num_images)
    test(num_images=num_images)