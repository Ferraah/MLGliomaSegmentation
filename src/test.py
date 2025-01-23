import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
import pytorch_lightning as pl
import yaml
from matplotlib.colors import ListedColormap



# Import the classes from your training script
from train import GliomaDataModule, GliomaSegmentation
import monai
from monai.metrics import DiceMetric

def load_splits():
    """Load the train/val/test splits from yaml file"""
    with open('splits-1.yaml', 'r') as f:
        splits = yaml.safe_load(f)
    return splits

def create_segmentation_colormap():
    """Create a custom colormap for segmentation visualization"""
    colors = [
        '#000000',  # Background - Black
        '#FF0000',  # NCR/NET - Red
        '#00FF00',  # ED - Green
        '#0000FF',  # ET - Blue
        '#FFFF00',  # Other - Yellow
    ]
    return ListedColormap(colors)

def visualize_results(model_path, data_module, num_samples=5, slice_idx=100):
    """
    Visualize results from the trained model.
    Args:
        model_path: Path to the trained model checkpoint
        data_module: GliomaDataModule instance
        num_samples: Number of samples to visualize
        slice_idx: Index of the slice to visualize (for 3D volumes)
    """
    # Load trained model
    model = GliomaSegmentation.load_from_checkpoint(model_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Get test dataloader
    # data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # Create output directory
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)

    # Create custom colormap
    seg_cmap = create_segmentation_colormap()

    with torch.no_grad():
        # Get samples
        for batch_idx, batch in enumerate(test_loader):

            # Move data to device
            images, labels = batch["image"], batch["label"]
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            # Get predictions
            outputs = model(images)

            # Convert to class indices
            pred_masks = torch.argmax(outputs, dim=1)
            true_masks = torch.argmax(labels, dim=1)

            # Compute Dice loss

            dice_metric = DiceMetric(include_background=True, reduction="mean")
            dice_metric(outputs, labels)
            dice_score = dice_metric.aggregate().item()
            print(f'Dice score for sample {batch_idx + 1}: {dice_score:.4f}')


            # Create figure for this sample
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Sample {batch_idx + 1}', fontsize=16)

            # Display T1 modality
            axes[0].imshow(images[0, 0, :, :, slice_idx].cpu().numpy(), cmap='gray')
            axes[0].set_title('T1 - Original')
            
            # Ground truth
            axes[1].imshow(
                true_masks[0, :, :, slice_idx].cpu().numpy(),
                cmap=seg_cmap,
                vmin=0,
                vmax=4
            )
            axes[1].set_title('Ground Truth')
            
            # Prediction
            axes[2].imshow(
                pred_masks[0, :, :, slice_idx].cpu().numpy(),
                cmap=seg_cmap,
                vmin=0,
                vmax=4
            )
            axes[2].set_title('Prediction')
            
            # Remove axes
            for ax in axes:
                ax.axis('off')
            
            # Add colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(cmap=seg_cmap, norm=plt.Normalize(vmin=0, vmax=4))
            cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(['Background', 'NCR/NET', 'ED', 'ET', 'Other'])
            
            # Save figure
            plt.savefig(
                output_dir / f'sample_{batch_idx + 1}.png',
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

            print(f'Saved visualization for sample {batch_idx + 1}')

def main():
    # Initialize data module with the test split
    data_module = GliomaDataModule(batch_size=1, channels=4)  # Use batch_size=1 for inference
    data_module.load_splits('splits/splits-4-training.yaml', 'splits/splits-4-validation.yaml', 'splits/splits-4-test.yaml', 'test')


    checkpoint_path = '/home/users/lguffanti/MLGliomaSegmentation/glioma_segmentation-epoch=21-val_loss=0.41.ckpt'  
    
    # Generate visualizations
    visualize_results(checkpoint_path, data_module)

if __name__ == "__main__":
    main()