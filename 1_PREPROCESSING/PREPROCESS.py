
import cv2
import numpy as np

class Preprocessor():
    def __init__(self, A_pth, B_pth, G_pth, R_pth):
        self.channels = {
            "A": cv2.imread(A_pth),
            "B": cv2.imread(B_pth),
            "G": cv2.imread(G_pth),
            "R": cv2.imread(R_pth)
        }
        self.processed_dict = {}

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l) 

        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def normalise_intensity(self, img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, woodchip_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
 
            bg_mask = cv2.bitwise_not(woodchip_mask)
            
            light_map = cv2.GaussianBlur(bg_mask, (65, 65), 0)
            light_map[light_map < 1] = 1
            
            mean_bg_val = cv2.mean(gray, mask=bg_mask)[0]
            light_3ch = cv2.merge([light_map, light_map, light_map])
            corrected_img = cv2.divide(img, light_3ch, scale=mean_bg_val)

            woodchip_mask_3ch = cv2.merge([woodchip_mask, woodchip_mask, woodchip_mask])
            
            final_output = np.where(woodchip_mask_3ch == 255, img, corrected_img)

            return final_output.astype(np.uint8)

    def process_clahe_intensity(self):
        for key, img in self.channels.items():
            proc = self.normalise_intensity(img)
            self.processed_dict[key] = self.apply_clahe(proc)

        return self.processed_dict

    def save_processed_images(self, num, pth_to_out):
        for key, img in self.processed_dict.items():
            cv2.imwrite(f"{pth_to_out}{num}_{key}.png", img)

    # def average_all_channels(self):
    #     layers = [self.processed_dict["A"], self.processed_dict["B"], 
    #               self.processed_dict["G"], self.processed_dict["R"]]

    #     stacked = np.array(layers, dtype=np.float32)
    #     avg_img = np.mean(stacked, axis=0)

    #     return avg_img.astype(np.uint8)

    # def save_average(self, num, pth_to_out):
    #     avg = self.average_all_channels()
    #     cv2.imwrite(f"{pth_to_out}{num}_Z.png", avg)

    def get_12_channel_stack(self):
        layers = [self.processed_dict["A"], self.processed_dict["B"], 
                  self.processed_dict["G"], self.processed_dict["R"]]

        return np.concatenate(layers, axis=-1)
        
    def get_local_rms(self):
        img = self.processed_dict["A"].astype(np.float32) / 255.0
        
        mu = cv2.blur(img, (11, 11))
        mu_sq = cv2.blur(cv2.multiply(img, img), (11, 11))
        
        sigma = cv2.sqrt(cv2.absdiff(mu_sq, cv2.multiply(mu, mu)))
        return sigma

    def get_stack(self):
        rms = self.get_local_rms()
        all_channels = self.get_12_channel_stack()

        final_stack = np.concatenate([all_channels, rms], axis=-1)
        
        return final_stack

    def save_multimodal_data(self, num, pth_to_out):
        stack = self.get_stack()
        out_name = f"{pth_to_out}{num}_multimodal.npz"
        np.savez_compressed(out_name, data=stack)


