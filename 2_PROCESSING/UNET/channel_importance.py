import itertools
import os
import numpy as np
import matplotlib.pyplot as plt

from PREDICT_FUNCTIONALITY import Predictor

CHECKPOINT = "./1/353ddf8fc5d6464ba39d04e019504093/checkpoints/epoch=49-step=450.ckpt"
PREDICT_DATA = "./2_PROCESSING/UNET/PREDICT/PREDICT_ARRAY/"

OUTPUT = "./3_POSTPROCESSING/OUT_UNET/"

preprocessed_images = sorted([f for f in os.listdir(PREDICT_DATA) if f.endswith(".npz")])
grouped_list = [(sample, list(images)) 
                for sample, images in itertools.groupby(preprocessed_images, 
                key=lambda x: x.split("_")[0])]

predict_image = Predictor(CHECKPOINT)
all_sample_sensitivities = []

for sample, images in sorted(grouped_list, key=lambda x: int(x[0])):
    print(f"Sample {sample}...")
    
    sample_scores = []
    
    for tile_name in images[:5]: 
        tile_path = os.path.join(PREDICT_DATA, tile_name)
        
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