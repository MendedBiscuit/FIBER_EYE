import cv2
import numpy as np


class Particle_Methods:
    def __init__(self, pth_to_img):
        self.img = cv2.imread(pth_to_img)

    def Watershed(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

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

    def Sobel_fill(self):
        # broken vibe code
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(sobelx, sobely)
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

        _, thresh = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((1, 1), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        mask = np.zeros_like(gray)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            if cv2.contourArea(cnt) > 0:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

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

        K = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv2.kmeans(
            data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape((blur.shape))

        return segmented_image

    def otsu(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        return otsu_img


class Impurity_Methods:
    def __init__(self, pth_to_img):
        self.img = cv2.imread(pth_to_img)

    def detect_impurities(self, wood_mask, sigma_thresh, mode="YUV"):
        if mode.upper() == "YUV":
            color_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
            c1, c2, c3 = cv2.split(color_img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))

            c1_p = clahe.apply(cv2.bitwise_and(c1, c1, mask=wood_mask))
            c2_p = clahe.apply(cv2.bitwise_and(c2, c2, mask=wood_mask))
            c3_p = clahe.apply(cv2.bitwise_and(c3, c3, mask=wood_mask))

        elif mode.upper() == "HSV":
            color_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            c1, c2, c3 = cv2.split(color_img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

            c1_p = cv2.bitwise_and(c1, c1, mask=wood_mask)
            c2_p = clahe.apply(cv2.bitwise_and(c2, c2, mask=wood_mask))
            c3_p = clahe.apply(cv2.bitwise_and(c3, c3, mask=wood_mask))

        else:
            raise ValueError("Mode must be 'YUV' or 'HSV'")

        def get_stats(channel):
            pixels = channel[channel > 0]
            if len(pixels) == 0:
                return 0, 0
            return np.mean(pixels), np.std(pixels)

        # m1, s1 = get_stats(c1_p)
        m2, s2 = get_stats(c2_p)
        m3, s3 = get_stats(c3_p)

        # anom1 = cv2.threshold(cv2.absdiff(c1_p, int(m1)), sigma_thresh * s1, 255, cv2.THRESH_BINARY)[1]
        anom2 = cv2.threshold(
            cv2.absdiff(c2_p, int(m2)), sigma_thresh * s2, 255, cv2.THRESH_BINARY
        )[1]
        anom3 = cv2.threshold(
            cv2.absdiff(c3_p, int(m3)), sigma_thresh * s3, 255, cv2.THRESH_BINARY
        )[1]

        # combined = cv2.bitwise_or(anom1, cv2.bitwise_or(anom2, anom3))
        combined = cv2.bitwise_or(anom2, anom3)

        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        impurity_mask = cv2.bitwise_and(combined, combined, mask=wood_mask)

        return impurity_mask

    def yuv_impurities(self, wood_mask):
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        y_eq = clahe.apply(y)
        u_eq = clahe.apply(u)
        v_eq = clahe.apply(v)

        def get_stats(channel, mask):
            pixels = channel[mask > 0]

            return np.mean(pixels), np.std(pixels)

        m_y, s_y = get_stats(y_eq, wood_mask)
        m_u, s_u = get_stats(u_eq, wood_mask)
        m_v, s_v = get_stats(v_eq, wood_mask)

        y_anom = cv2.threshold(
            cv2.absdiff(y_eq, int(m_y)), 2 * s_y, 255, cv2.THRESH_BINARY
        )[1]
        u_anom = cv2.threshold(
            cv2.absdiff(u_eq, int(m_u)), 2 * s_u, 255, cv2.THRESH_BINARY
        )[1]
        v_anom = cv2.threshold(
            cv2.absdiff(v_eq, int(m_v)), 2 * s_v, 255, cv2.THRESH_BINARY
        )[1]

        combined = cv2.bitwise_or(y_anom, cv2.bitwise_or(u_anom, v_anom))
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        impurity_mask = cv2.bitwise_and(combined, combined, mask=wood_mask)

        return impurity_mask

    def hsv_impurities(self, wood_mask):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        s_eq = clahe.apply(cv2.bitwise_and(s, s, mask=wood_mask))
        v_eq = clahe.apply(cv2.bitwise_and(v, v, mask=wood_mask))
        h_masked = cv2.bitwise_and(h, h, mask=wood_mask)

        def get_stats(channel):
            pixels = channel[channel > 0]
            if len(pixels) == 0:
                return 0, 0

            return np.mean(pixels), np.std(pixels)

        m_h, s_h = get_stats(h_masked)
        m_s, s_s = get_stats(s_eq)
        m_v, s_v = get_stats(v_eq)

        h_anom = cv2.threshold(
            cv2.absdiff(h_masked, int(m_h)), 2 * s_h, 255, cv2.THRESH_BINARY
        )[1]
        s_anom = cv2.threshold(
            cv2.absdiff(s_eq, int(m_s)), 2 * s_s, 255, cv2.THRESH_BINARY
        )[1]
        v_anom = cv2.threshold(
            cv2.absdiff(v_eq, int(m_v)), 2 * s_v, 255, cv2.THRESH_BINARY
        )[1]

        combined = cv2.bitwise_or(h_anom, cv2.bitwise_or(s_anom, v_anom))
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        impurity_mask = cv2.bitwise_and(combined, combined, mask=wood_mask)

        return impurity_mask
