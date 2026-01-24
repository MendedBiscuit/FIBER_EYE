import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import config as c
import numpy as np
from tqdm import tqdm

def calculate_stats(image_dir, channels=15):
    files = [f for f in os.listdir(image_dir) if f.endswith('.npz')]
    
    psum = np.zeros(channels)
    psum_sq = np.zeros(channels)
    pixel_count = 0

    for f in tqdm(files):
        img = np.load(os.path.join(image_dir, f))["image"] 
        
        pixel_count += img.shape[0] * img.shape[1]
 
        psum += np.sum(img, axis=(0, 1))
        psum_sq += np.sum(np.square(img), axis=(0, 1))

    mean = psum / pixel_count
    std = np.sqrt((psum_sq / pixel_count) - np.square(mean))
    
    return mean, std
