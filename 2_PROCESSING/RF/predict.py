import cv2

from FOREST_FUNCTIONALITY import WoodchipRF

rf_handler = WoodchipRF()
rf_handler.load("./2_PROCESSING/RF/model/rf_woodchip_model.joblib") 

test_tile = "./2_PROCESSING/UNET/PREDICT/PREDICT_ARRAY/2_y0_x0_multimodal.npz"

mask_result = rf_handler.predict_tile(test_tile)

cv2.imwrite("./2_PROCESSING/RF/pred.png", mask_result * 255)