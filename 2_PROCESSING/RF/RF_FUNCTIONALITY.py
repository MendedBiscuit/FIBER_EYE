import os
import cv2
import numpy as np
import joblib


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


class WoodchipRF:
    def __init__(self, n_estimators=256, max_depth=8):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=50,       
            max_features=None,          
            class_weight="balanced",
            n_jobs=-1,
            verbose=1,
        )

    def _prep_data(self, image_dir, mask_dir, num_samples=10, subsample_rate=0.1):
        X_list, y_list = [], []
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".npz")])[:num_samples]
        
        for f in image_files:
            data = np.load(os.path.join(image_dir, f))
            img = data["image"]
            
            mask_filename = f.replace(".npz", ".png")
            mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_GRAYSCALE)

            pixels = img.reshape(-1, 15)
            labels = mask.reshape(-1)

            indices = np.random.choice(
                len(labels), int(len(labels) * subsample_rate), replace=False
            )
            
            X_list.append(pixels[indices])
            y_list.append(labels[indices])

        return np.vstack(X_list), np.concatenate(y_list)

    def train(self, image_dir, mask_dir, num_samples=10):
        X, y = self._prep_data(image_dir, mask_dir, num_samples=num_samples)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)

        val_acc = self.model.score(X_val, y_val)

    def predict_image(self, img_15ch):
        h, w, c = img_15ch.shape
        flat_img = img_15ch.reshape(-1, 15)
        preds = self.model.predict(flat_img)
        return preds.reshape(h, w)

    def save(self, path):
        joblib.dump(self.model, path)

    def tune_parameters(self, image_dir, mask_dir, num_samples=5, subsample_rate=0.01):

        X, y = self._prep_data(image_dir, mask_dir, num_samples=num_samples, subsample_rate=subsample_rate)
        
        param_grid = {
            "n_estimators": [128, 256, 512],
            "max_depth": [8, 16, 32, 64],
            "min_samples_leaf": [1, 12, 18, 24, 32, 48],
            "max_features": [2, 3, 6, 12, 15],
            "bootstrap": [True, False]
        }

        print(f"Grid Search {X.shape[0]} samples")
        grid_search = GridSearchCV(
            RandomForestClassifier(class_weight="balanced"), 
            param_grid=param_grid, 
            cv=3, 
            n_jobs=-1, 
            verbose=2
        )
        
        grid_search.fit(X, y)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return grid_search.best_params_