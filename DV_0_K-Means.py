# K-Means Quantization for set of Images
# Ondrej Fercak, 7/22/2023

import numpy as np
import cv2
import math
import glob
from PIL import Image

# INPUT
K = 18          # Number of Quantized Bins (i.e. How Many Colors)
MX = 30         # Termination Criteria: Max Iterations
E = 0.5         # Termination Criteria: Epsilon (Quality)


######################################################################################
# DEFINE FUNCTIONS
######################################################################################
def k_means_visual(cnt, unq):
    # Establish Color Bar Size
    bar_height = 900
    bar_width = 450

    # Concatenate & Sort Data
    cnt = np.reshape(cnt, (len(cnt), 1))
    color_dat = np.hstack((cnt, unq))
    color_sorted = np.flipud(color_dat[color_dat[:, 0].argsort()])
    color_index = np.linspace(0, len(cnt) - 1, len(cnt)).astype(int).reshape(len(cnt), 1)
    color_sorted = np.hstack((color_index, color_sorted))

    # Initialize Empty Color Bar & Height Object
    color_bar = np.array([], np.uint8).reshape((0, bar_width, 3))
    height_increase = []

    # Create & Append Each Row's Results to Color Bar
    for row in color_sorted:
        row_percent = 1/len(color_sorted[:, 1])
        row_height = math.floor(bar_height * row_percent)
        row_canvas = np.ones((int(row_height), int(bar_width), 3), np.uint8)
        row_color = row[2:5].reshape(1, 3)
        row_final = np.where(row_canvas == row_canvas, row_color, 0).astype(np.uint8)
        color_bar = cv2.vconcat([color_bar, row_final])
        height_increase.append(row_height)
        cv2.putText(color_bar, str(row[0]), (7, int(np.sum(height_increase) - row_height + 20)),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(color_bar, str(np.fliplr(row_color)[0]), (int(width/5) - 150, int(np.sum(height_increase) - row_height/2) + 9),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    return color_sorted, color_bar


######################################################################################
# INPUT PARAMETERS/ INITIALIZE
######################################################################################
path = "***Your Path Here***"

for file in list(glob.glob(path + '/*.jpg')):
    img = cv2.imread(file)

    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    size = 1
    shape = img.shape

    height = shape[0]
    width = shape[1]
    image_resize = cv2

    # Create a white image, a window
    canvas_red = np.ones((int(height), int(width), 3), np.uint8) * 255
    canvas_blue = np.ones((int(height), int(width), 3), np.uint8) * 255
    mask_red = np.zeros((int(height), int(width), 3), np.uint8)
    mask_blue = np.zeros((int(height), int(width), 3), np.uint8)

    ######################################################################################
    # MAIN LOOP
    ######################################################################################
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MX, E/10)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, MX, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    unique, counts = np.unique(res, return_counts=True, axis=0)

    color_data, color_display = k_means_visual(counts, unique)
    cv2.imwrite(file + "_K-MEANS [RGB].tiff", color_display)
    data = Image.open(file + "_K-MEANS [RGB].tiff")
    data.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
