import time
from monai.config import print_config
from monai.losses import DiceLoss

from dataloader import GliomaDataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from pytorch_lightning import Trainer

class GliomaSegmentationModel(pl.LightningModule):
    def __init__(self, max_epochs):
        super(GliomaSegmentationModel, self).__init__()
        self.learning_rate = 1e-4
        self.loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        self.metrics = {"Dice": DiceMetric(include_background=False, reduction="mean")}
        self.max_epochs = max_epochs

        self.model = UNet(
            spatial_dims=3,     # 3D image
            in_channels=1,      # One grayscale channel
            out_channels=5,     # Segmentation mask with 4 classes plus background
            channels=(32, 64, 128, 256), # Number of channels in each layer
            strides=(2, 2, 2), # Strides in each layer
            num_res_units=2,  # Number of residual units
            norm=Norm.BATCH   # Batch normalization
        )

        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        
        dice_score = self.metrics["Dice"](outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        dice_score = self.metrics["Dice"].aggregate().item()
        self.metrics["Dice"].reset()
        self.log('val_dice', dice_score, on_step=False, on_epoch=True, prog_bar=True)
        # Save the best model based on validation dice score
        if not hasattr(self, 'best_val_dice') or dice_score > self.best_val_dice:
            self.best_val_dice = dice_score
            torch.save(self.model.state_dict(), 'best_model.pth')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 1e-1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]

def train(max_epochs=2, num_images=10):
    print("[TRAIN] Num available threads: ", torch.get_num_threads())

    training_loader, validation_loader, testing_loader = GliomaDataLoader.get_loaders(num_images=num_images)

    model = GliomaSegmentationModel(max_epochs=max_epochs)

    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(model, training_loader, validation_loader)
