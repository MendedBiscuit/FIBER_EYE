
import cv2
import numpy as np

class Preprocess_Colour():
    def __init__(self, A_pth, b_pth, g_pth, r_pth):
        self.channels = {
            "A": cv2.imread(A_pth),
            "B": cv2.imread(b_pth),
            "G": cv2.imread(g_pth),
            "R": cv2.imread(r_pth)
        }
        self.processed_dict = {}

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        enhanced = cv2.merge((l, a, b))

        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def normalise_intensity(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        light_map = cv2.GaussianBlur(gray, (129, 129), 0)
        light_map[light_map < 1] = 1
        
        mask_indices = mask > 0

        mean_val = np.mean(img[mask_indices])
            
        light_3ch = cv2.merge([light_map, light_map, light_map])
        corrected = cv2.divide(img, light_3ch, scale=mean_val)

        return cv2.bitwise_and(corrected, corrected, mask=mask)

    def process_all(self):
        for key, img in self.channels.items():
            proc = self.normalise_intensity(img)
            self.processed_dict[key] = self.apply_clahe(proc)

        return self.processed_dict

    def get_12_channel_stack(self):
        layers = [self.processed_dict["A"], self.processed_dict["B"], 
                  self.processed_dict["G"], self.processed_dict["R"]]

        return np.concatenate(layers, axis=-1)

    def average_all_channels(self):
        layers = [self.processed_dict["A"], self.processed_dict["B"], 
                  self.processed_dict["G"], self.processed_dict["R"]]

        stacked = np.array(layers, dtype=np.float32)
        avg_img = np.mean(stacked, axis=0)

        return avg_img.astype(np.uint8)

    def save_processed_images(self, num, pth_to_out):
        for key, img in self.processed_dict.items():
            cv2.imwrite(f"{pth_to_out}{num}_{key}.png", img)

    def save_average(self, num, pth_to_out):
        avg = self.average_all_channels()
        cv2.imwrite(f"{pth_to_out}{num}_Z.png", avg)