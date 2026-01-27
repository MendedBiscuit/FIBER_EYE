import os
import re
import torch

import numpy as np

from ultralytics import YOLO


class InstancePredictor:
    def __init__(self, checkpoint_path):
        self.model = YOLO(checkpoint_path)

    def predict_tile_instances(self, pth_to_tile, x_offset, y_offset):
        results = self.model.predict(pth_to_tile, retina_masks=True, verbose=False)
        res = results[0]
        
        tile_instances = []
        if res.masks is not None:
            for i, poly in enumerate(res.masks.xy):
                global_poly = poly.copy()
                global_poly[:, 0] += x_offset
                global_poly[:, 1] += y_offset
                
                tile_instances.append({
                    "polygon": global_poly.astype(np.int32),
                    "class": int(res.boxes.cls[i]),
                    "conf": float(res.boxes.conf[i])
                })
        return tile_instances

    def stitch_to_instance_map(self, directory_path, sample_tiles, tile_size=512):
        all_chips = []
        
        for f in sample_tiles:
            match = re.search(r"y(\d+)_x(\d+)", f)
            if match:
                y_off, x_off = int(match.group(1)), int(match.group(2))
                tile_path = os.path.join(directory_path, f)
                chips = self.predict_tile_instances(tile_path, x_off, y_off)
                all_chips.extend(chips)

        max_y = max([np.max(c["polygon"][:, 1]) for c in all_chips]) + 10
        max_x = max([np.max(c["polygon"][:, 0]) for c in all_chips]) + 10

        instance_mask = np.zeros((int(max_y), int(max_x)), dtype=np.int32)

        for idx, chip in enumerate(all_chips):
            unique_id = idx + 1
            cv2.fillPoly(instance_mask, [chip["polygon"]], unique_id)

        return instance_mask, all_chips

    def visualize_instances(self, instance_mask):
        max_id = np.max(instance_mask)
        colors = np.random.randint(0, 255, size=(max_id + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        
        colored_mask = colors[instance_mask]
        return colored_mask.astype(np.uint8)