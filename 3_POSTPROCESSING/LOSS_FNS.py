import cv2
import numpy as np
import segmentation_models_pytorch as smp

class Calculate_Metrics():
    def __init__(self, pth_to_pred, pth_to_mask):
        self.pred = cv2.imread(pth_to_pred)
        self.mask = cv2.imread(pth_to_mask)
