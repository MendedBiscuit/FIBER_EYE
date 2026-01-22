import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import itertools

import config as c
import numpy as np
from CV_FUNCTIONALITY import Particle_Methods, Impurity_Methods

cv_img = sorted([f for f in os.listdir(c.CV_IN) if f.endswith(".png")])
GROUPS = itertools.groupby(cv_img, key=lambda x: (x.split("_")[0], x.split("_")[1]))

def ensemble_impurities(masks, threshold=3):
    ensemble_sum = np.sum(masks, axis=0).astype(np.float32)

    final_mask = np.where(ensemble_sum >= threshold * 255, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    return final_mask, ensemble_sum

for sample, images in GROUPS:
    if sample == "1":
        particle_mask = Particle_Methods(c.CV_IN + str(sample) + "_A.png")
        mask = particle_mask.otsu()

        masks_list = []

        for image in images:
            print(image)
            process_image = Impurity_Methods(c.CV_IN + image)

            masks_list.append(process_image.detect_impurities(mask, 2, mode="YUV"))
            masks_list.append(process_image.detect_impurities(mask, 2, mode="HSV"))

            # img = process_image.detect_impurities(mask, 2, mode="YUV")
            # img2 = process_image.detect_impurities(mask, 3, mode="HSV")

            # cv2.imshow("z", img)
            # cv2.imshow("x", img2)
            # cv2.waitKey(0)

            # masks_list.append(process_image.yuv_impurities(mask))
            # masks_list.append(process_image.hsv_impurities(mask))

        ensemble_stack = np.array(masks_list)

        final_mask, _ = ensemble_impurities(ensemble_stack, threshold=5)
        cv2.imshow("test", final_mask)
        cv2.waitKey(0)
