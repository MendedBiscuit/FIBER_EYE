import os
import torch
import mlflow
import pytorch_lightning as L
import albumentations as A

from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from model import UNet
from dataset import SpanData

# Train Parameters

EPOCHS = 256
BATCH_SIZE = 32

TRAIN_DATA = "./2_PROCESSING/UNET/TRAINING_ARRAY/"
TRAIN_MASK = "./2_PROCESSING/_MASKS/TRAIN_MASK/"
VALID_DATA = "./2_PROCESSING/UNET/VALID_ARRAY/"
VALID_MASK = "./2_PROCESSING/_MASKS/VALID_MASK/"

TRAIN_TRANSFORM = A.Compose(
    [
        A.Resize(544, 544),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

VALID_TRANSFORM = A.Compose(
    [
        A.Resize(544, 544),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def rename_model_file():
    """
    Renames the model file to "model.ckpt"
    """
    version = max([int(x.split("_")[-1]) for x in os.listdir("./lightning_logs/")])
    directory = f"./lightning_logs/version_{version}/checkpoints/"
    old = os.path.join(directory, f"epoch={EPOCHS - 1}-step={EPOCHS * 2}.ckpt")
    new = os.path.join(directory, "model.ckpt")
    os.rename(old, new)


def main():
    """
    Functionality for running the training
    """

    # Prepare training and validation data with appropriate transforms
    train_ds = SpanDataset(TRAIN_IMG, TRAIN_MASK, transform=TRAIN_TRANSFORM)
    val_ds = SpanDataset(VALID_DATA, VALID_MASK, transform=VALID_TRANSFORM)

    # Load data for training, change num_workers to increase CPU/GPU load
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )

    # Configure model
    model = UNet(encoder_name="resnet34", in_channels=12, classes=2, t_max=EPOCHS)

    # mlflow logger
    mlflow.pytorch.autolog(log_models=True)
    mlf_logger = MLFlowLogger(
        experiment_name="Spunet_Segmentation_Run", tracking_uri="file:./ml-runs"
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        precision="16-mixed",
    )

    # Lightning module responsible for running the training loop
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    rename_model_file()


main()
