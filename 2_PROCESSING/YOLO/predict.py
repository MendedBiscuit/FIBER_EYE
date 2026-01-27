import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import torch
import itertools

import config as c

from ultralytics import YOLO
from PREDICT_FUNCTIONALITY import Predictor

preprocessed_img = sorted(
    [f for f in os.listdir(c.YOLO_PREDICT_IMG) if f.endswith(".png")]
)
grouped_list = [
    (sample, list(images))
    for sample, images in itertools.groupby(
        preprocessed_img, key=lambda x: x.split("_")[0]
    )
]

predict_image = Predictor(c.YOLO_MODEL)

for sample, images in sorted(grouped_list, key=lambda x: int(x[0])):
    prediction = predict_image.stitch_tiles(
        c.YOLO_PREDICT_IMG, images, tile_size=c.TILE_SIZE, stride=c.STRIDE
    )

    cv2.imwrite(os.path.join(c.YOLO_OUT, f"{sample}_P_YOLO.png"), prediction * 255)
