import cv2
import numpy as np

# img = cv2.imread("1_PREPROCESSING/IN_PREPROCESS/1_A.png")
img = cv2.imread("2_CLASSIC_CV/OUT_PREPROCESS/1_A.png")

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Convert the tuple to a list
channels = list(cv2.split(img_yuv))
# channels = list(cv2.split(img_hsv))

# Now this assignment works
channels[2] = cv2.equalizeHist(channels[2])

img_yuv = cv2.merge(channels)
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imwrite("1at.png", img_output)
cv2.imshow("a", img_output)
cv2.waitKey(0)

#RED_image -> H
#YUV -> all