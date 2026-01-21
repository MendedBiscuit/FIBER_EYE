import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score

class WoodchipRF:
    def __init__(self, n_estimators=100, max_depth=15):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            n_jobs=-1, # all CPU cores
            verbose=1
        )

    def _prep_data(self, image_dir, mask_dir, num_samples=10):
        """Extracts pixels from tiles to create a training table."""
        X_list, y_list = [], []
        image_files = sorted(os.listdir(image_dir))[:num_samples]

        for f in image_files:
            img = np.load(os.path.join(image_dir, f))["image"]
            mask_filename = f.replace(".npz", ".png")
            
            mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_GRAYSCALE)

            X_list.append(img.reshape(-1, 15))
            y_list.append(mask.reshape(-1))

        return np.vstack(X_list), np.concatenate(y_list)

    def train(self, image_dir, mask_dir):
        X, y = self._prep_data(image_dir, mask_dir)
        
        self.model.fit(X, y)
        print("Training complete.")

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def predict_tile(self, tile_path):
        """Predicts a single tile and reshapes it back to an image."""
        img = np.load(tile_path)["image"]
        h, w, c = img.shape
        
        flat_img = img.reshape(-1, c)
        
        preds = self.model.predict(flat_img)

        return preds.reshape(h, w).astype(np.uint8)
