import torch
import pytorch_lightning as pl
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from GliomaDataModule import GliomaDataModule



class GliomaSegmentation(pl.LightningModule):
    def __init__(self, channels=1, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
        # Update U-Net to handle 4 input channels (T1ce, T2, FLAIR, T1)
        self.model = UNet(
            spatial_dims=3,
            in_channels=channels,  # Changed from 1 to 4 to handle all modalities
            out_channels=5,  # 4 classes (background + 3 tumor regions)
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
        
        print(f"Model initialized with {channels} input channels")

        # Loss function and metrics remain the same
        self.loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
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
        # Compute Dice score
        self.dice_metric(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log('val_dice', dice_score, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        self.dice_metric(outputs, labels)
        return outputs

    def on_test_epoch_end(self):
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log('test_dice', dice_score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
