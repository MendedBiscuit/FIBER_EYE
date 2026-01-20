import os
import cv2
import itertools

from pycocotools.coco import COCO

from PREPROCESS import Preprocessor


IN_PREPROCESS = "./1_PREPROCESSING/IN_PREPROCESS/IN_DATA/"
RAW_MASKS = "./1_PREPROCESSING/IN_PREPROCESS/mask_data.json" # specify .json file for raw mask data

PNG_MASKS = "./1_PREPROCESSING/IN_PREPROCESS/MASK_PNG" # directory for masks as .png 

OUT_PREPROCESS = "./2_PROCESSING/CLASSIC_CV/IN_CV/" # used for classical CV methods

# Output of the chopped up data for UNET training
OUT_TRAIN_DATA = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_ARRAY/" # this directory will have arrays
OUT_MASKS = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_MASK/" # and this one .png 

PREPROCESSED_IMG = sorted([f for f in os.listdir(IN_PREPROCESS) if f.endswith('.png')])
groups = itertools.groupby(PREPROCESSED_IMG, key=lambda x: x.split("_")[0])

for sample, images in groups:
    A_B_G_R_list = sorted(list(images))
    colour_paths = [os.path.join(IN_PREPROCESS, x) for x in A_B_G_R_list]

    processor = Preprocessor(
        colour_paths[0], colour_paths[1], colour_paths[2], colour_paths[3]
    )

    # Apply normalisation and such
    processor.process_clahe_intensity()

    # Save the images normally
    processor.save_processed_images(sample, OUT_PREPROCESS)    

    # Get the .png version of each mask
    processor.masks_from_json(RAW_MASKS, PNG_MASKS)

    channel_array = processor.get_stack()
    mask = cv2.imread(f"./1_PREPROCESSING/IN_PREPROCESS/MASK_PNG/{sample}_M.png")

    processor.tile_and_save(channel_array, mask, sample, OUT_TRAIN_DATA, OUT_MASKS, tile_size=512, stride=256)


