import os
import cv2
import itertools
from CV_FUNCTIONALITY import Particle_Methods

IN_CV = "./2_CLASSIC_CV/OUT_PREPROCESS/"
OUT_CV = "./2_CLASSIC_CV/OUT_CV/"

CV_IMG = sorted([f for f in os.listdir(IN_CV) if f.endswith(".png")])
GROUPS = itertools.groupby(CV_IMG, key=lambda x: x.split("_")[0])

for sample, images in GROUPS:
    if sample == "1":
        A_B_G_R_list = sorted(list(images))
        colour_paths = [os.path.join(IN_CV, x) for x in A_B_G_R_list]
        print(colour_paths)
        image = Methods(colour_paths[0])

        test = image.Watershed()
        # test = image.Canny(50, 150)
        # test = image.Sobel()
        # test = image.Laplacian()
        # test = image.k_means()


        cv2.imshow("a", test)

        cv2.waitKey(0)

# aa = Methods("./1_PREPROCESSING/IN_PREPROCESS/1_A.png")
# bb = Methods("./2_CLASSIC_CV/OUT_PREPROCESS/1_R.png")

# # test = aa.Watershed()
# # test = aa.Canny(50, 150)
# # test = aa.Sobel()
# # test = aa.Laplacian()
# # test = aa.k_means()

# # testa = bb.Watershed()
# # testa = bb.Canny(50, 150)
# # testa = bb.Sobel_fill()
# # testa = bb.Sobel()
# # testa = bb.Laplacian()
# # testa = bb.k_means()
# testa = bb.otsu()

# # cv2.imshow("a", test)
# cv2.imshow("b", testa)
# cv2.waitKey(0)

# # want -> particle analysis using sobel? or otsu? configure watershed
# # separate -> impurity analysis, odd threshold