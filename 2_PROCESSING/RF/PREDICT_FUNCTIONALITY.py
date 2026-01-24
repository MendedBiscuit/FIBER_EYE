import os
import re
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, pth_to_model):
        self.model = joblib.load(pth_to_model)

    def stitch_tiles(self, directory_path, sample_tiles, tile_size=512, stride=256):
        all_coords = []
        for f in sample_tiles:
            match = re.search(r"y(\d+)_x(\d+)", f)
            if match:
                y, x = int(match.group(1)), int(match.group(2))
                all_coords.append((f, y, x))

        max_y = max(c[1] for c in all_coords) + tile_size
        max_x = max(c[2] for c in all_coords) + tile_size

        full_prob_map = np.zeros((max_y, max_x), dtype=np.float32)
        count_mask = np.zeros((max_y, max_x), dtype=np.float32)

        for fname, y, x in all_coords:
            tile_path = os.path.join(directory_path, fname)

            mask_tile = self.predict_tile(tile_path)

            full_prob_map[y : y + tile_size, x : x + tile_size] += mask_tile
            count_mask[y : y + tile_size, x : x + tile_size] += 1

        final_mask = full_prob_map / np.maximum(count_mask, 1)

        return (final_mask > 0.8).astype(np.uint8)

    def predict_tile(self, tile_path):
        img = np.load(tile_path)["image"]
        h, w, c = img.shape

        flat_img = img.reshape(-1, c)

        preds = self.model.predict(flat_img)

        return preds.reshape(h, w).astype(np.uint8)

    def plot_feature_importance(self, channel_names=None):
        if channel_names is None:
            channel_names = [f"Channel {i}" for i in range(15)]

        importances = self.model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 6))
        plt.title("Importance of Each Input Channel")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [channel_names[i] for i in indices])
        plt.xlabel("Relative Importance Score")
        plt.show()
