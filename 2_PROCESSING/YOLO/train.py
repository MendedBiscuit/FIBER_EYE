import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml
import torch

import config as c
import torch.nn as nn

from ultralytics import YOLO

def start_training():
    dataset_config = {
        "train": c.YOLO_TRAIN_IMG,
        "val": c.YOLO_VALID_IMG,
        "names": {0: "bg", 1: "WoodChip"}
    }

    with open('data.yaml', 'w') as f:
        yaml.dump(dataset_config, f)

    model = YOLO(c.BASE_YOLO_MODEL) 
    model.train(
        data='data.yaml',  
        imgsz=544,            
        batch=16, 
        device=0,
        workers=6 
    )

if __name__ == '__main__':
    start_training()