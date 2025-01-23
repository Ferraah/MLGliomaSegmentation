import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from GliomaDataModule import GliomaDataModule
from model import GliomaSegmentation



def train_model(channels=1):
    # Initialize data module and model
    data_module = GliomaDataModule(batch_size=4, channels=channels)  # Reduced batch size due to 3D data
    model = GliomaSegmentation(channels=channels)

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints-{channels}-weighted',
        filename='glioma_segmentation-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=-1,
        every_n_epochs=2
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        strategy='ddp',
        logger=True
    )

    # Train model
    trainer.fit(model, data_module)


if __name__ == '__main__':
    train_model(channels=4)