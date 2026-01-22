import os
import cv2
import torch
import itertools

from model import UNet
from PREDICT_FUNCTIONALITY import Predictor

# Predict Parameters

CHECKPOINT = "./1/353ddf8fc5d6464ba39d04e019504093/checkpoints/epoch=49-step=450.ckpt"
PREDICT_DATA = "./2_PROCESSING/UNET/PREDICT/PREDICT_ARRAY/"

OUTPUT = "./3_POSTPROCESSING/OUT_UNET/"

preprocessed_images = sorted([f for f in os.listdir(PREDICT_DATA) if f.endswith(".npz")])
grouped_list = [(sample, list(images)) 
                for sample, images in itertools.groupby(preprocessed_images, 
                key=lambda x: x.split("_")[0])]

predict_image = Predictor(CHECKPOINT)

for sample, images in sorted(grouped_list, key=lambda x: int(x[0])):

    prediction = predict_image.stitch_tiles(PREDICT_DATA, images, tile_size=512, stride=256)

    cv2.imwrite(OUTPUT + f"{sample}_P_UNET.png", prediction * 255)

