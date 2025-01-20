

from test import test
from train import train

if __name__ == "__main__":
    train(max_epochs=20, num_images=100)
    test(num_images=100)