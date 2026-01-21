
from FOREST_FUNCTIONALITY import WoodchipRF


TRAIN_IMG = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_ARRAY/"
TRAIN_MSK = "./2_PROCESSING/UNET/IN_UNET/TRAIN/TRAIN_MASK/"

rf_handler = WoodchipRF()

rf_handler.train(TRAIN_IMG, TRAIN_MSK)
rf_handler.save("./2_PROCESSING/RF/model/rf_woodchip_model.joblib")

