import cv2
import numpy as np

IMG_PTH = "./data/IN_PREPROCESS/"

def normalise_intensity(pth_to_img):
        print(pth_to_img)
        img = cv2.imread(pth_to_img)
        gray_amb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        light_map = cv2.GaussianBlur(gray_amb, (33, 33), 0)
        light_map[light_map < 1] = 1

        mean_val = np.mean(img)
        light_3ch = cv2.merge([light_map, light_map, light_map])

        return cv2.divide(img, light_3ch, scale=mean_val)

# cv2.imwrite("1r.png", normalise_intensity(IMG_PTH + "1r.png"))

def extract_visual_channel(img, color="R"):

    result = img.copy()

    if color == "R":
        result[:, :, 0] = 0 
        result[:, :, 1] = 0 
    elif color == "G":
        result[:, :, 0] = 0 
        result[:, :, 2] = 0 
    elif color == "B":
        result[:, :, 1] = 0 
        result[:, :, 2] = 0 
        
    return result

def rev_extract_visual_channel(img, color="R"):
    if color == "B":
        img[:, :, 0] = 0 
    elif color == "G":
        img[:, :, 1] = 0 
    elif color == "R":
        img[:, :, 2] = 0 

    return img


cv2.imwrite("1nogreen.png", rev_extract_visual_channel(normalise_intensity(IMG_PTH + "1g.png"), "G"))