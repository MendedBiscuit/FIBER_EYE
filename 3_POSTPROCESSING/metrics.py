import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import config as c
import itertools

from collections import defaultdict
from METRIC_FUNCTIONALITY import Calculate_Metrics

def render_table(data):
    methods = list(data.keys())
    metric_names = list(data[methods[0]].keys())

    header = f"{'Method':<10} | " + " | ".join([f"{name:<10}" for name in metric_names])
    
    print("\nBinary segmentation of impurities ")
    print("-" * (len(header)//2 + 3
    ))
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<10} | "
        row += " | ".join([f"{data[method][m]:<10.4f}" for m in metric_names])
        print(row)


OUTPUTS = [(c.CV_OUT, "CV"), (c.RF_OUT, "RF"), (c.UNET_OUT, "UNET")] 

masks= sorted(
    [f for f in os.listdir(c.PNG_MASKS) if f.endswith(".png")]
)

summary_table = {}

for method_path, method_name in OUTPUTS:
    predictions = [f for f in os.listdir(method_path) if f.endswith(".png")]
    sorted_preds = sorted(predictions, key=lambda x: int(x.split("_")[0]))

    totals = defaultdict(float)
    count = 0

    for sample in sorted_preds:
        sample_id = sample.split("_")[0]
        pred_path = os.path.join(method_path, sample)
        mask_path = os.path.join(c.PNG_MASKS, f"{sample_id}_M.png")

        if not os.path.exists(mask_path):
            continue

        evaluate = Calculate_Metrics(pred_path, mask_path)
        print(pred_path)
        metrics = evaluate.get_metrics() # (name, value)

        for name, value in metrics:
            totals[name] += value
        count += 1

    if count > 0:
        summary_table[method_name] = {name: (val / count) for name, val in totals.items()}

render_table(summary_table)
