import os
import itertools
import re
from PREPROCESS_COLOUR import Preprocess_Colour
from STATS import Contrast

IN_PREPROCESS = "./data/IN_PREPROCESS/"
OUT_PREPROCESS = "./data/OUT_PREPROCESS/"
PRE_PROCESS_DATA = os.listdir(IN_PREPROCESS)

groups = itertools.groupby(PRE_PROCESS_DATA, key=lambda x: x.split(".")[0][:-1])

for sample, images in groups:
    AMB_B_G_R_list = list(images)

    for i in range(0, len(AMB_B_G_R_list)):
        colour_paths = [IN_PREPROCESS + x for x in AMB_B_G_R_list]

        processor = Preprocess_Colour(
            colour_paths[0], colour_paths[1], colour_paths[2], colour_paths[3]
        )

        processor.output_preprocessed(i + 1, OUT_PREPROCESS)
