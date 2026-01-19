import cv2
import numpy as np


class Methods():
    def __init__(self, pth_to_img):
        self.img = cv2.imread(pth_to_img)

    def Watershed(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(self.img, markers)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        mask[markers > 1] = 255
        
        return mask

    def Canny(self, min_val, max_val):
        # gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(self.img, (5, 5), 1.4)
        edges = cv2.Canny(self.img, min_val, max_val) 
        
        return edges
    
    def Sobel(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) 
        
        gradient_magnitude = cv2.magnitude(sobelx, sobely)
        
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

        return gradient_magnitude

    def Laplacian(self):
        # gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        laplace = cv2.Laplacian(self.img, ddepth, ksize=3)
        laplace = cv2.convertScaleAbs(laplace)
        
        return laplace

    def k_means(self):
        # img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        blur = cv2.GaussianBlur(self.img, (5, 5), 1.4)
        data = blur.reshape((-1, 3))
        data = np.float32(data)

        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape((blur.shape))
        
        return segmented_image

    def otsu(self):
        
        return