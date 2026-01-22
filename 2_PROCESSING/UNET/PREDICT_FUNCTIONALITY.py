
import torch
import os
import re
import numpy as np

from model import UNet


class Predictor():
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_tile(self, pth_to_tile):

        data = np.load(pth_to_tile)['image'] 
        data = data.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            # pr_mask = (logits.sigmoid() > 0.5).float()
            pr_mask = logits.sigmoid()

        return pr_mask.squeeze().cpu().numpy()

    def stitch_tiles(self, directory_path, sample_tiles, tile_size=512, stride=256):

            all_coords = []
            for f in sample_tiles:
                match = re.search(r'y(\d+)_x(\d+)', f)
                if match:
                    y, x = int(match.group(1)), int(match.group(2))
                    all_coords.append((f, y, x))

            max_y = max(c[1] for c in all_coords) + tile_size
            max_x = max(c[2] for c in all_coords) + tile_size

            full_prob_map = np.zeros((max_y, max_x), dtype=np.float32)
            count_mask = np.zeros((max_y, max_x), dtype=np.float32)
            
            print(f"{max_y}x{max_x}")

            for fname, y, x in all_coords:
                tile_path = os.path.join(directory_path, fname)
                
                mask_tile = self.predict_tile(tile_path)
                
                full_prob_map[y:y+tile_size, x:x+tile_size] += mask_tile
                count_mask[y:y+tile_size, x:x+tile_size] += 1

            final_mask = full_prob_map / np.maximum(count_mask, 1)


            return (final_mask > 0.5).astype(np.uint8)
    
    def get_channel_sensitivity(self, pth_to_tile):
            """Calculates the MSE impact of shuffling each channel on a single tile."""
            data = np.load(pth_to_tile)['image'] # (512, 512, 15)
            if data.shape[-1] == 15:
                data = data.transpose(2, 0, 1) # (15, 512, 512)
            
            orig_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                baseline = torch.sigmoid(self.model(orig_tensor)).cpu().numpy()

            sensitivities = []
            for c in range(15):
                perturbed_data = data.copy()
                c_slice = perturbed_data[c].flatten()
                np.random.shuffle(c_slice)
                perturbed_data[c] = c_slice.reshape(512, 512)
                
                pert_tensor = torch.from_numpy(perturbed_data).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    perturbed_out = torch.sigmoid(self.model(pert_tensor)).cpu().numpy()
                
                mse = np.mean((baseline - perturbed_out)**2)
                sensitivities.append(mse)
                
            return np.array(sensitivities)