import yaml
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, "nomenclature.yaml"), "r") as f:
    cfg = yaml.safe_load(f)


def get_path(*keys):
    try:
        temp = cfg
        for k in keys:
            temp = temp[k]

        if isinstance(temp, str):
            return os.path.abspath(os.path.join(ROOT_DIR, temp))

        return temp
    except (KeyError, TypeError):
        return None


# paths

IN_TRAIN_DATA = get_path("paths", "preprocessing", "training_data")
IN_PREDICT_DATA = get_path("paths", "preprocessing", "prediction_data")

RAW_MASKS = get_path("paths", "preprocessing", "json_mask")
PNG_MASKS = get_path("paths", "preprocessing", "png_masks")

IMG_PNG = get_path("paths", "processing", "data", "input", "tiled_images")

UNET_MODEL = get_path("paths", "processing", "unet", "model")
RF_MODEL = get_path("paths", "processing", "rf", "model")

CV_OUT = get_path("paths", "processing", "classic_cv", "output")
UNET_OUT = get_path("paths", "processing", "unet", "output")
RF_OUT = get_path("paths", "processing", "rf", "output")

TRAIN_ARRAY = get_path("paths", "processing", "data", "input", "train_array")
TRAIN_MASK = get_path("paths", "processing", "data", "input", "train_mask")
VALID_ARRAY = get_path("paths", "processing", "data", "input", "validation_array")
VALID_MASK = get_path("paths", "processing", "data", "input", "validation_mask")
PREDICT_ARRAY = get_path("paths", "processing", "data", "input", "prediction_array")

# vars
TILE_SIZE = get_path("important_variables", "tile_size")
STRIDE = get_path("important_variables", "stride")

CHANNEL_LABELS = [
    "Ambient Blue",
    "Ambient Green",
    "Ambient Red",
    "BlueLED Blue",
    "BlueLED Green",
    "BlueLED Red",
    "GreenLED Blue",
    "GreenLED Green",
    "GreenLED Red",
    "RedLED Blue",
    "RedLED Green",
    "RedLED Red",
    "RMS Texture 1",
    "RMS Texture 2",
    "RMS Texture 3",
]
