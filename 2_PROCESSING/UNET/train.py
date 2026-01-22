import os
import torch
import mlflow
import pytorch_lightning as L
import albumentations as A

from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from model import UNet
from dataset import SpanDataset

# Train Parameters

EPOCHS = 50
BATCH_SIZE = 8

TRAIN_ARRAY = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_ARRAY"
TRAIN_MASK = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_MASK"
VALID_ARRAY = "./2_PROCESSING/UNET/IN_UNET/VALID/VALID_ARRAY"
VALID_MASK = "./2_PROCESSING/UNET/IN_UNET/VALID/VALID_MASK"

TRAIN_TRANSFORM = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)}, 
        scale=(0.9, 1.1), 
        rotate=(-45, 45), 
        p=0.5
    ),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2), 
    A.GaussNoise(std_dev=(10.0, 50.0), p=0.2),
    ToTensorV2(),
])

VALID_TRANSFORM = A.Compose([
    ToTensorV2(),
])


# def rename_model_file():
#     """
#     Renames the model file to "model.ckpt"
#     """
#     version = max([int(x.split("_")[-1]) for x in os.listdir("./lightning_logs/")])
#     directory = f"./lightning_logs/version_{version}/checkpoints/"
#     old = os.path.join(directory, f"epoch={EPOCHS - 1}-step={EPOCHS * 2}.ckpt")
#     new = os.path.join(directory, "model.ckpt")
#     os.rename(old, new)


def main():
    """
    Functionality for running the training
    """

    # Prepare training and validation data with appropriate transforms
    train_ds = SpanDataset(TRAIN_ARRAY, TRAIN_MASK, transform=TRAIN_TRANSFORM)
    val_ds = SpanDataset(VALID_ARRAY, VALID_MASK, transform=VALID_TRANSFORM)

    # Load data for training, change num_workers to increase CPU/GPU load
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False
    )

    # Configure model
    model = UNet(encoder_name="resnet34", in_channels=15, classes=1, t_max=EPOCHS)

    # mlflow logger
    mlflow_logger = L.loggers.MLFlowLogger(
        experiment_name="Woodchip_Impurity_Detection",
        tracking_uri="sqlite:///mlflow.db"
    )

    trainer = L.Trainer(
        logger=mlflow_logger,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        precision="16-mixed",
    )

    # Lightning module responsible for running the training loop
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # rename_model_file()

if __name__ == '__main__':
    main()