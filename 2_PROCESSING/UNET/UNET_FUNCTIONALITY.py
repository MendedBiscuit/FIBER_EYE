import torch
import os
import cv2

import numpy as np
import torch.nn as nn
import albumentations as A
import pytorch_lightning as L
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from albumentations.pytorch import ToTensorV2


class UNet(L.LightningModule):
    """
    Configuration of UNet and training loop

    -- Parameters --
    encoder_name : str
    in_channels : int, number of channels in input images
    classes : int, number of output classes (leave at 1 for binary segmentation)
    t_max : int, learning rate decay

    """

    def __init__(
        self, encoder_name="resnet34", in_channels=15, classes=1, t_max=64, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set model parameters
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=4,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            **kwargs,
        )

        means = [6.82410381e+01, 6.76335912e+01, 6.54805091e+01, 1.12030697e+02,
                5.29831737e+01, 2.74933674e+01, 8.63749225e+01, 1.27529567e+02,
                4.93052940e+01, 3.00348355e+01, 4.05368050e+01, 1.24959776e+02,
                9.92451729e-02, 1.01227612e-01, 1.03890658e-01]

        stds =[71.01557075, 73.78593944, 76.9774136,  87.64590613, 67.35343469, 44.09452226,
                66.45241989, 71.36465389, 68.49253075, 42.0748052,  48.7104614,  68.77368723,
                0.09902525,  0.10010371,  0.10150998]
        
        mean_tensor = torch.tensor(calculated_means).view(1, in_channels, 1, 1).float()
        std_tensor = torch.tensor(calculated_stds).view(1, in_channels, 1, 1).float()

        # 3. Register as buffersw
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

        # Tversky and Focal loss
        # 1. Setup the Tversky Loss
        self.tversky_loss = smp.losses.TverskyLoss(mode="binary", alpha=0.7, beta=0.5)  
        self.focal_loss = smp.losses.FocalLoss(mode="binary")

        # Prepare for training outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure mask and image have the same shape [B, in_channels, H, L]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Calculate loss
        mask = (mask > 0).float()
        logits_mask = self.forward(image)

        # Loss type
        loss = self.tversky_loss(logits_mask, mask) + 5 * self.focal_loss(
            logits_mask, mask
        )

        prob_mask = logits_mask.sigmoid()
        # pred_mask = logits_mask.argmax(dim=1)
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        step_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            f"{stage}_iou_step", step_iou, prog_bar=True, on_step=True, on_epoch=False
        )

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def training_step(self, batch, batch_idx):
        # Prepare training data
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        # Prepare validation data
        out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out["loss"]

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)

    def on_train_epoch_end(self):
        # Clear epoch training data
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        # Clear epoch validation data
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Learning rate and parameters for the UNET
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.t_max, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class SpanDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Prepare training images and masks for loading

        -- Parameters --
        img_dir : str
        mask_dir : str
        transform : Albumentations sequential
        """

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".npz")])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.load(img_path)["image"].astype(np.float32)

        mask_name = img_name.replace(".npz", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()
