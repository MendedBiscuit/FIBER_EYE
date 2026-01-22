import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import itertools

import config as c
import numpy as np
import matplotlib.pyplot as plt

from PREDICT_FUNCTIONALITY import Predictor

preprocessed_images = sorted([f for f in os.listdir(c.PREDICT_ARRAY) if f.endswith(".npz")])
grouped_list = [(sample, list(images)) 
                for sample, images in itertools.groupby(preprocessed_images, 
                key=lambda x: x.split("_")[0])]

predict_image = Predictor(c.UNET_MODEL)
all_sample_sensitivities = []

for sample, images in sorted(grouped_list, key=lambda x: int(x[0])):
    print(f"Sample {sample}...")
    
    sample_scores = []
    
    for tile_name in images[:5]: 
        tile_path = os.path.join(c.PREDICT_ARRAY, tile_name)
        
        scores = predict_image.get_channel_sensitivity(tile_path)
        sample_scores.append(scores)
        
    all_sample_sensitivities.append(np.mean(sample_scores, axis=0))

global_importance = np.mean(all_sample_sensitivities, axis=0)

def plot_importance(importance_scores):
    plt.figure(figsize=(12, 6))
    channels = [f"Ch {i}" for i in range(15)]

    norm_importance = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores) + 1e-8)
    
    plt.bar(channels, norm_importance, color='teal', edgecolor='black', alpha=0.8)
    plt.title("Combined Channel Sensitivity (aggregated across all tiles)", fontsize=14)
    plt.ylabel("Relative Importance (Normalized MSE)", fontsize=12)
    plt.xlabel("Spectral Channel Index", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

plot_importance(global_importance)