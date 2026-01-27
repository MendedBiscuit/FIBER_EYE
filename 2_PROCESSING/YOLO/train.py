import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import config as c
import torch.nn as nn

from ultralytics import YOLO

dataset_config = {
    'path': '/home/julian/Data/fibereye/datasets/woodchips_tiled', # Base directory
    'train': 'images',  # Folder containing tiled .png files
    'val': 'images',    # Folder for validation
    'names': {0: 'WoodChip'}
}

model = YOLO("yolo26n-seg.yaml") 

model.train(
    data="./2_PROCESSING/YOLO/yolo_config.yaml",   
    imgsz=544,            
    batch=16, 
    device=0,               
)