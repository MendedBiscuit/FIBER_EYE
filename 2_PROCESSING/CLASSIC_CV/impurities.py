import os
import cv2
import itertools
import numpy as np
from CV_FUNCTIONALITY import Particle_Methods, Impurity_Methods

IN_CV = "./2_PROCESSING/CLASSIC_CV/IN_CV/"
OUT_CV = "./3_POSTPROCESSING/OUT_CV/"

CV_IMG = sorted([f for f in os.listdir(IN_CV) if f.endswith(".png")])
GROUPS = itertools.groupby(CV_IMG, key=lambda x: (x.split("_")[0], x.split("_")[1]))

def ensemble_impurities(masks, threshold=3):
    ensemble_sum = np.sum(masks, axis=0).astype(np.float32)

    final_mask = np.where(ensemble_sum >= threshold * 255, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    return final_mask, ensemble_sum

for sample, images in GROUPS:
    if sample == "1":
        particle_mask = Particle_Methods(IN_CV + str(sample) + "_A.png")
        mask = particle_mask.otsu()

        masks_list = []

        for image in images:
            print(image)
            process_image = Impurity_Methods(IN_CV + image)

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

        final_mask, summy = ensemble_impurities(ensemble_stack, threshold=5)
        print(summy)
        cv2.imshow("test", final_mask)
        cv2.waitKey(0)
