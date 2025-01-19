

from test import test
from train import train

if __name__ == "__main__":
    train(max_epochs=2)
    test()