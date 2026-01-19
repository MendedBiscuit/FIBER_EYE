import os
import itertools
from PREPROCESS import Preprocessor


IN_PREPROCESS = "./1_PREPROCESSING/IN_PREPROCESS/"
OUT_PREPROCESS = "./2_PROCESSING/CLASSIC_CV/OUT_PREPROCESS/"
TRAINING_DATA = "./2_PREPROCESSING/UNET/TRAINING_ARRAY/"

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
    processor.save_multimodal_data(sample, TRAINING_DATA)
