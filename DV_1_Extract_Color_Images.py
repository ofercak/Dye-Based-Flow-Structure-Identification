# K-Means Quantization for set of Images & Image Extraction
# Ondrej Fercak, 7/22/2023

import numpy as np
import cv2
import math
from PIL import Image
import glob

# INPUT
K = 18
MX = 30
E = 0.5

color_range = 7

# Color Data [RGB Colors Generated from K-Means]
color_red = np.array([[148, 104, 112],
                      [143, 70, 73],
                      [160, 133, 141],
                      [129, 31, 24]])

color_blue = np.array([[116, 140, 153],
                       [99, 110, 121],
                       [43, 61, 64]])


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

cnt = 0

frames = np.linspace(1, 500, 500).astype(int)
for cnt in frames:
    file = path + '/{}.jpg'.format(cnt)
    print(file)
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
    cv2.imwrite(path + "/" + file + "_K-MEANS [RGB]-Dark.tiff", color_display)
    data = Image.open(path + "/" + file + "_K-MEANS [RGB]-Dark.tiff")
    data.show()

    test = input('Go?')

    kmn = res.reshape(img.shape)
    compare = cv2.hconcat((img, kmn))
    cv2.imwrite(path + "/" + file + "_K_Means_Image.tiff", kmn)

    for i, row_red in enumerate(color_red):
        color = row_red.reshape(1, 3)
        up = np.fliplr(color)[0] + color_range
        lw = np.fliplr(color)[0] - color_range
        image_red = cv2.inRange(kmn, lw, up)
        image_red = cv2.cvtColor(image_red, cv2.COLOR_GRAY2RGB)
        canvas_red[:, :, :] = np.where(image_red == (255, 255, 255), img, canvas_red)
        mask_red[:, :, :] = np.where(image_red == (255, 255, 255), (255, 255, 255), mask_red)
        image_red_show = cv2.resize(canvas_red, (int(width * 0.1), int(height * 0.1)), interpolation=cv2.INTER_AREA)
        cv2.imshow('image_red{}'.format(i), image_red_show)

    for j, row_blue in enumerate(color_blue):
        color = row_blue.reshape(1, 3)
        up = np.fliplr(color)[0] + color_range * 2
        lw = np.fliplr(color)[0] - color_range * 2
        image_blue = cv2.inRange(kmn, lw, up)
        image_blue = cv2.cvtColor(image_blue, cv2.COLOR_GRAY2RGB)
        canvas_blue[:, :, :] = np.where(image_blue == (255, 255, 255), img, canvas_blue)
        mask_blue[:, :, :] = np.where(image_blue == (255, 255, 255), (255, 255, 255), mask_blue)
        image_blue_show = cv2.resize(canvas_blue, (int(width * 0.1), int(height * 0.1)), interpolation=cv2.INTER_AREA)
        cv2.imshow('image_blue{}'.format(j), image_blue_show)

    final = cv2.hconcat((compare, canvas_red))
    final = cv2.hconcat((final, canvas_blue))
    final = cv2.resize(final, (int(4 * width * size), int(height * size)), interpolation=cv2.INTER_AREA)
    cv2.imshow('Color', final)
    cv2.imwrite(file + "_All.tiff", final)

    canvas_red = cv2.medianBlur(canvas_red, 3)
    canvas_red = cv2.resize(canvas_red, (int(width * size), int(height * size)), interpolation=cv2.INTER_AREA)
    cv2.imshow('Red', canvas_red)
    cv2.imwrite(file + "_Red.tiff", canvas_red)
    cv2.imwrite(file + "_Mask_Red.tiff", mask_red)

    canvas_blue = cv2.medianBlur(canvas_blue, 3)
    canvas_blue = cv2.resize(canvas_blue, (int(width * size), int(height * size)), interpolation=cv2.INTER_AREA)
    cv2.imshow('Blue', canvas_blue)
    cv2.imwrite(file + "_Blue.tiff", canvas_blue)
    cv2.imwrite(file + "_Mask_Blue.tiff", mask_blue)

cv2.waitKey(0)
cv2.destroyAllWindows()
