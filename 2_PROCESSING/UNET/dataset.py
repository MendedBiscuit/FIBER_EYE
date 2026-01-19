import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SpanDataset(Dataset):
    """
    Prepare training images and masks for loading

    -- Parameters --
    img_dir : str
    mask_dir : str
    transform : Albumentations sequential 

    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.images[idx])
            image = np.load(img_path)['image'].astype(np.float32)

            mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.npz', '.png'))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            #change mask logic
            mask = mask.astype(np.int64) 

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                norm = A.Compose([
                    A.Resize(544, 544),
                    ToTensorV2(),
                ])
                augmented = norm(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
        
            return image, mask
