import os
import cv2
import itertools

from pycocotools.coco import COCO

from PREPROCESS import Preprocessor

TRAIN_VALID_THRESH = 8/10 # Portion of images used as training data

# Settings for tiling the images
TILE_SIZE = 1024
STRIDE = 1

IN_PREPROCESS = "./1_PREPROCESSING/IN_PREPROCESS/IN_DATA/"
RAW_MASKS = "./1_PREPROCESSING/IN_PREPROCESS/mask_data.json"  # specify .json file for raw mask data

PNG_MASKS = "./1_PREPROCESSING/IN_PREPROCESS/MASK_PNG"  # directory for masks as .png

OUT_PREPROCESS = "./2_PROCESSING/CLASSIC_CV/IN_CV/"  # used for classical CV methods

# Output of the chopped up data for UNET training
OUT_TRAIN_ARRAY = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_ARRAY/"  # this directory will have arrays
OUT_TRAIN_MASK = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_MASK/"  # and this one .png

OUT_VALID_ARRAY = "./2_PROCESSING/UNET/IN_UNET/VALID/VALID_ARRAY/"  # this directory will have arrays
OUT_VALID_MASK = "./2_PROCESSING/UNET/IN_UNET/VALID/VALID_MASK/"  # and this one .png

PREPROCESSED_IMG = sorted([f for f in os.listdir(IN_PREPROCESS) if f.endswith(".png")])
groups = itertools.groupby(PREPROCESSED_IMG, key=lambda x: x.split("_")[0])

for sample, images in sorted(groups, key=lambda x: int(x[0])):

    A_B_G_R_list = sorted(list(images))
    colour_paths = [os.path.join(IN_PREPROCESS, x) for x in A_B_G_R_list]

    processor = Preprocessor(
                    colour_paths[0],
                    colour_paths[1],
                    colour_paths[2], 
                    colour_paths[3]
                )

    # Apply normalisation and such
    processor.process_clahe_intensity()

    # Save tiled images for CV methods
    processor. CV_tile_and_save(sample, OUT_PREPROCESS, tile_size=1024, stride=1)

    # Get the .png version of each mask
    processor.masks_from_json(RAW_MASKS, PNG_MASKS)

    channel_array = processor.get_stack()
    mask = cv2.imread(f"./1_PREPROCESSING/IN_PREPROCESS/MASK_PNG/{sample}_M.png")

    # Tile UNET training data
    if int(sample) <= int(TRAIN_VALID_THRESH * len(list(groups))):
        processor.UNET_tile_and_save(
            sample,
            channel_array,
            mask,
            OUT_TRAIN_ARRAY,
            OUT_TRAIN_MASK,
            tile_size=512,
            stride=256,
        )
    
    else:
        processor.UNET_tile_and_save(
            sample,
            channel_array,
            mask,
            OUT_VALID_ARRAY,
            OUT_VALID_MASK,
            tile_size=TILE_SIZE,
            stride=STRIDE,
        )

