from monai.config import print_config
from monai.losses import DiceLoss

from dataloader import GliomaDataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
from model_trainer import ModelTrainer
from monai.metrics import DiceMetric


def test(num_images=10):

    print("[TEST] Num available threads: ", torch.get_num_threads())

    _, _, testing_loader = GliomaDataLoader.get_loaders(num_images=num_images)

    device = torch.device("cuda")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    ).to(device)

    max_epochs = 2

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-1, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler(enabled=False)
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

    # Evaluate on validation/test data
    print("[TEST] ",trainer.evaluate(testing_loader))


if __name__ == "__main__":
    test()