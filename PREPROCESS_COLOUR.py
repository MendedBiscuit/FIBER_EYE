
import cv2
import numpy as np

class Preprocess_Colour():
    def __init__(self, amb_pth, b_pth, g_pth, r_pth):
        self.channels = {
                        "Amb": cv2.imread(amb_pth),
                          "B": cv2.imread(b_pth),
                          "G": cv2.imread(g_pth) ,
                          "R": cv2.imread(r_pth) }

    def normalise_intensity(self):
        gray_amb = cv2.cvtColor(self.channels["Amb"], cv2.COLOR_BGR2GRAY)
        light_map = cv2.GaussianBlur(gray_amb, (65, 65), 0)
        light_map[light_map < 1] = 1

        mean_val = np.mean(self.channels["Amb"])
        light_3ch = cv2.merge([light_map, light_map, light_map])

        corrected = {}
        for key in self.channels.keys():
            corrected[key] = cv2.divide(self.channels[key], 
                                        light_3ch, scale=mean_val
                                        )
        
        return corrected

    def restrict_channel(img, channel=None):
        if channel == "R":
            img[:, :, 0] = 0 
            img[:, :, 1] = 0 
        elif channel == "G":
            img[:, :, 0] = 0 
            img[:, :, 2] = 0 
        elif channel == "B":
            img[:, :, 1] = 0 
            img[:, :, 2] = 0 
        
        return img

    def combined_image():
        return

    def output_preprocessed(self, num, pth_to_out, intensity_norm=False, channel_restrict=False):
        corrected = self.normalise_intensity()

        for key in corrected.keys():
            binary_norm = self.one_channel(key, corrected[key])
            out_name = pth_to_out + str(num) + key + ".png"
            cv2.imwrite(out_name, binary_norm)

# normalise + 