import os 
import itertools
import numpy as np
from STATS import Contrast

PREPROCESSED_DIR = "./data/OUT_PREPROCESS/"
PREPROCESSED_IMG = sorted([f for f in os.listdir(PREPROCESSED_DIR) if f.endswith('.png')])

groups = itertools.groupby(PREPROCESSED_IMG, key=lambda x: x.split("_")[0])

best_contrast = []

for sample, images in groups:
    results = []
    for img in images:
        processor = Contrast(PREPROCESSED_DIR + img)

        Michel = processor.Michelson()
        # RMS = processor.RMS()

        results.append((img, Michel))

    best_contrast.append(max(results, key=lambda x: x[1]))

print(best_contrast)