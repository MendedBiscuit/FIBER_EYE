import os
import torch
import cv2
import numpy as np
import albumentations as A

from torch.utils.data import Dataset

from albumentations.pytorch import ToTensorV2


class SpunetDataset(Dataset):
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

        mask_viz = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = np.zeros_like(mask_viz, dtype=np.int64)
        # mask[mask_viz == 127] = 1  # wood_chip
        mask[mask_viz == 255] = 1  # impurity

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            to_tensor = A.Compose(
                [
                    A.Resize(544, 544),
                    ToTensorV2(),
                ]
            )
            augmented = to_tensor(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()
