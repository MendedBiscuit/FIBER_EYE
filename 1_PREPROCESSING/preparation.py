import os
import itertools

from pycocotools.coco import COCO

from PREPROCESS import Preprocessor


IN_PREPROCESS = "./1_PREPROCESSING/IN_PREPROCESS/IN_DATA/"
MASKS = "./1_PREPROCESSING/IN_PREPROCESS/IN_MASK/mask_data.json" # specify .json file for masks

OUT_PREPROCESS = "./2_PROCESSING/CLASSIC_CV/IN_CV/"

OUT_TRAIN_DATA = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_ARRAY/"
OUT_MASKS = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_MASK/"


PREPROCESSED_IMG = sorted([f for f in os.listdir(IN_PREPROCESS) if f.endswith('.png')])
groups = itertools.groupby(PREPROCESSED_IMG, key=lambda x: x.split("_")[0])

for sample, images in groups:
    A_B_G_R_list = sorted(list(images))
    colour_paths = [os.path.join(IN_PREPROCESS, x) for x in A_B_G_R_list]

    processor = Preprocessor(
        colour_paths[0], colour_paths[1], colour_paths[2], colour_paths[3]
    )

    processor.process_clahe_intensity()
    processor.save_processed_images(sample, OUT_PREPROCESS)
    processor.save_multimodal_data(sample, OUT_TRAIN_DATA)

    processor.masks_from_json(MASKS, OUT_MASKS)


