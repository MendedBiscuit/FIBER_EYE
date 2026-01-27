import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config as c

from RF_FUNCTIONALITY import WoodchipRF

model = WoodchipRF()

best_params = model.tune_parameters(c.TRAIN_ARRAY, c.TRAIN_MASK)

model.train(c.TRAIN_ARRAY, c.TRAIN_MASK, num_samples=20)

model.save(c.RF_MODEL)