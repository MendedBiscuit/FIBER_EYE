import cv2
import numpy as np

class Contrast:
    def __init__(self, pth_to_img):
        self.pth_to_img = pth_to_img
        self.img = cv2.imread(self.pth_to_img)

    def Michelson(self):
        Y = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)[:, :, 0]

        min_Y = float(np.min(Y))
        max_Y = float(np.max(Y))

        contrast = (max_Y - min_Y) / (max_Y + min_Y)
        return contrast

    def RMS(self):
        img = self.img.astype(float) / 255.0
        mean, std_dev = cv2.meanStdDev(img)
        rms_contrast = std_dev[0][0]

        return rms_contrast

    def Local_RMS(self):
        img = self.img.astype(float) / 255.0

        mu = cv2.blur(img, (11, 11))
        mu_sq = cv2.blur(img**2, (11, 11))
        return np.sqrt(np.abs(mu_sq - mu**2))

    def Intesity_Range(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        return img.std()
