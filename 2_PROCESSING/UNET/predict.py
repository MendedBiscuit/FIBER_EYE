import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import torch
import itertools

import config as c

from preparation import TILE_SIZE, STRIDE
from model import UNet
from PREDICT_FUNCTIONALITY import Predictor

preprocessed_array = sorted([f for f in os.listdir(c.PREDICT_ARRAY) if f.endswith(".npz")])
grouped_list = [(sample, list(images)) 
                for sample, images in itertools.groupby(preprocessed_array, 
                key=lambda x: x.split("_")[0])]

predict_image = Predictor(CHECKPOINT)

for sample, images in sorted(grouped_list, key=lambda x: int(x[0])):

    prediction = predict_image.stitch_tiles(c.PREDICT_ARRAY, images, tile_size=c.TILE_SIZE, stride=c.STRIDE)

    cv2.imwrite(c.UNET_OUT + f"{sample}_P_UNET.png", prediction * 255)

