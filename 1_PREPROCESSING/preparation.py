import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import config as c
import itertools

from pycocotools.coco import COCO

from PREPROCESS import Preprocessor

TRAIN_VALID_THRESH = 8/10 # Portion of images used as training data

# Get the .png version of each mask
processor.masks_from_json(c.RAW_MASKS, c.PNG_MASKS)

# Sort samples in directory IN_PROCESS
preprocessed_images = sorted([f for f in os.listdir(c.IN_TRAIN_DATA) if f.endswith(".png")])
grouped_list = [(sample, list(images)) 
                for sample, images in itertools.groupby(preprocessed_images, 
                key=lambda x: x.split("_")[0])]

num_samples = len(grouped_list)

for sample, images in sorted(grouped_list, key=lambda x: int(x[0])):
    A_B_G_R_list = sorted(list(images))
    colour_paths = [os.path.join(c.IN_TRAIN_DATA, x) for x in A_B_G_R_list]

    processor = Preprocessor(
                    colour_paths[0],
                    colour_paths[1],
                    colour_paths[2], 
                    colour_paths[3]
                )

    # Apply normalisation and such
    processor.process_clahe_intensity()

    # Save tiled images for CV methods
    processor.CV_tile_and_save(sample, c.CV_IN, tile_size=c.TILE_SIZE, stride=c.STRIDE)

    channel_array = processor.get_stack()
    mask = cv2.imread(f"{c.PNG_MASKS}{sample}_M.png")

    # Tile UNET training data
    if int(sample) <= int(TRAIN_VALID_THRESH * num_samples):
        processor.UNET_tile_and_save(
            sample,
            channel_array,
            mask,
            c.TRAIN_ARRAY,
            c.TRAIN_MASK,
            tile_size=c.TILE_SIZE,
            stride=c.STRIDE,
        )
    
    else:
        processor.UNET_tile_and_save(
            sample,
            channel_array,
            mask,
            c.VALID_ARRAY,
            c.VALID_MASK,
            tile_size=c.TILE_SIZE,
            stride=c.STRIDE,
        )

