import os
import itertools
from PREPROCESS import Preprocessor


IN_PREPROCESS = "./1_PREPROCESSING/IN_PREPROCESS/"
OUT_PREPROCESS = "./2_CLASSIC_CV/OUT_PREPROCESS/"
TRAINING_DATA = "./3_UNET/TRAINING_ARRAY/"

PREPROCESS_DATA = sorted(os.listdir(IN_PREPROCESS))

groups = itertools.groupby(PREPROCESS_DATA, key=lambda x: x.split(".")[0][:-1])

# Use when naming convention is new with _
# PREPROCESSED_IMG = sorted([f for f in os.listdir(PREPROCESSED_DIR) if f.endswith('.png')])
# groups = itertools.groupby(PREPROCESSED_IMG, key=lambda x: x.split("_")[0])

for sample, images in groups:

    A_B_G_R_list = sorted(list(images))
    colour_paths = [os.path.join(IN_PREPROCESS, x) for x in A_B_G_R_list]

    processor = Preprocessor(
        colour_paths[0], 
        colour_paths[1], 
        colour_paths[2], 
        colour_paths[3]
    )

    processor.process_clahe_intensity()
    processor.save_processed_images(sample, OUT_PREPROCESS)
    processor.save_multimodal_data(sample, TRAINING_DATA)
