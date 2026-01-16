import os
import itertools
import re
from PREPROCESS_COLOUR import Preprocess_Colour

IN_PREPROCESS = "./data/IN_PREPROCESS/"
OUT_PREPROCESS = "./data/OUT_PREPROCESS/"
PREPROCESS_DATA = sorted(os.listdir(IN_PREPROCESS))

groups = itertools.groupby(PREPROCESS_DATA, key=lambda x: x.split(".")[0][:-1])

# Use when naming convention is new with _
# PREPROCESSED_IMG = sorted([f for f in os.listdir(PREPROCESSED_DIR) if f.endswith('.png')])
# groups = itertools.groupby(PREPROCESSED_IMG, key=lambda x: x.split("_")[0])

for sample, images in groups:

    AMB_B_G_R_list = sorted(list(images))
    colour_paths = [os.path.join(IN_PREPROCESS, x) for x in AMB_B_G_R_list]

    if len(colour_paths) == 4:
        processor = Preprocess_Colour(
            colour_paths[0], 
            colour_paths[1], 
            colour_paths[2], 
            colour_paths[3]
        )

        processor.process_all()
        processor.save_processed_images(sample, OUT_PREPROCESS)
        processor.save_average(sample, OUT_PREPROCESS)

        # stack = processor.get_12_channel_stack()
