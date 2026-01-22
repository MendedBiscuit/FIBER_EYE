import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config as c

from FOREST_FUNCTIONALITY import WoodchipRF

TRAIN_IMG = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_ARRAY/"
TRAIN_MSK = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_MASK/"

rf_handler = WoodchipRF()

rf_handler.train(c.TRAIN_ARRAY, c.TRAIN_MASK)
rf_handler.save(c.RF_MODEL)

