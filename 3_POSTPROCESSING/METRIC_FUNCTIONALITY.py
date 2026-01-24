import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score

class Calculate_Metrics:
    def __init__(self, pth_to_pred, pth_to_mask):

        pred_img = cv2.imread(pth_to_pred, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(pth_to_mask, cv2.IMREAD_GRAYSCALE)
    
        self.pred = (pred_img > 127).astype(np.uint8)
        self.mask = (mask_img > 127).astype(np.uint8)

        self.tp = np.logical_and(self.pred == 1, self.mask == 1).sum()
        self.fp = np.logical_and(self.pred == 1, self.mask == 0).sum()
        self.fn = np.logical_and(self.pred == 0, self.mask == 1).sum()
        self.tn = np.logical_and(self.pred == 0, self.mask == 0).sum()
    
    def iou(self):
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        if union == 0: return 1.0
        return intersection / union

    def precision(self):
        if (self.tp + self.fp) == 0: return 0.0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if (self.tp + self.fn) == 0: return 0.0
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p + r) == 0: return 0.0
        return (2 * (p * r) / (p + r), p, r)

    def get_metrics(self):
        f1, prec, rec = self.f1_score()
        return [
                ("IOU", float(self.iou())),
                ("Recall", float(rec)),
                ("Precision", float(prec)),
                ("F1", float(f1))
               ]