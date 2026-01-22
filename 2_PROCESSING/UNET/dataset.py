import os
import torch
import cv2
import numpy as np
import albumentations as A

from torch.utils.data import Dataset

from albumentations.pytorch import ToTensorV2


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
        else:
            continue

        return image, mask.long()
