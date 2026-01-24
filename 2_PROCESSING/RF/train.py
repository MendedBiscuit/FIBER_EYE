import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config as c

from RF_FUNCTIONALITY import WoodchipRF

rf_handler = WoodchipRF()

rf_handler.train(c.TRAIN_ARRAY, c.TRAIN_MASK)
rf_handler.save(c.RF_MODEL)
