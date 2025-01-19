import time
from monai.config import print_config
from monai.losses import DiceLoss

from dataloader import GliomaDataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
from model_trainer import ModelTrainer
from monai.metrics import DiceMetric


def train(max_epochs=2, num_images=10):
    print("[TRAIN] Num available threads: ", torch.get_num_threads())

    training_loader, validation_loader, testing_loader = GliomaDataLoader.get_loaders(num_images=num_images)

    device = torch.device("cpu")
    model = UNet(
        spatial_dims=3,     # 3D image
        in_channels=1,      # One grayscale channel
        out_channels=5,     # Segmetation mask with 5 classes
        channels=(32, 64, 128, 256), # Number of channels in each layer
        strides=(2, 2, 2), # Strides in each layer
        num_res_units=2,  # Number of residual units
        norm=Norm.BATCH   # Batch normalization
    ).to(device)


    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-1, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True
    metrics = {"Dice": DiceMetric(include_background=False)}

    # Using a warpper to train model and print metrics
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_function,
        scheduler=lr_scheduler,
        metrics=metrics,
    )

    history = trainer.train(
        train_loader=training_loader,
        val_loader=validation_loader,
        epochs=max_epochs,
        log_interval=10,
        save_path="best_model.pth",
    )

    print(history)



if __name__ == "__main__":
    train()
    