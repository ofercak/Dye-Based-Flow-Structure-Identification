# Vortex Tracking Algorithm [Direct Numerical Simulation Parameters]
# Ondrej Fercak, 7/22/2023

import cv2
import numpy as np
import csv
import time
import os

area_min = 50
area_max = 1800
v_line = 38
h_line = 113
canvas = np.ones((228, 512), np.uint8) * 255


def vortex_contours(img):
    disp_color = (0, 0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('Image Size: {}'.format(np.shape(gray)))

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    element = 3
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (element, element))
    cv2.imshow('Thesh1', threshold)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, elem, iterations=2)
    cv2.imshow('Thesh2', threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # list for storing names of shapes
    blobs = []
    contours_new = []
    for j, cnt1 in enumerate(contours):
        blob_area = cv2.contourArea(cnt1)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt1)
        if area_min < blob_area < area_max:
            print('Blob Area Keep = {}/{}/{}'.format(j, blob_area, np.shape(cnt1)))
            blobs.append(blob_area)
            contours_new.append(cnt1)

    if len(blobs) > 0:
        rev_sort = np.linspace(len(blobs) - 1, 0, len(blobs)).astype(int)

        blobs2 = np.array(blobs)
        contours2 = np.array(contours)

        blobs22 = blobs2[rev_sort]
        contours22 = contours2[rev_sort]

        blobs3 = blobs22
        contours3 = contours22

        centers = []
        for cnt3 in contours3:
            (cx3, cy3), radius = cv2.minEnclosingCircle(cnt3)
            blob_area3 = cv2.contourArea(cnt3)
            centers.append((int(cx3), int(cy3)))
            cv2.circle(img, (int(cx3), int(cy3)), int(radius), disp_color, 2)
    else:
        print("Nada".format())
        centers = []

    return img, np.array(centers)


####################################################################################################
# MAIN
####################################################################################################

path = "***Your Path Here***"
save_path = "***Your Path Here***"

file_csv = save_path + "/Vortex Distances.csv"
header = ["Image #", "Color", "X-Y Locations"]

with open(file_csv, 'x', newline='') as f:
    writer = csv.writer(f, delimiter=",", skipinitialspace=True)
    writer.writerow(header)
    time.sleep(1)
    f.close()

frames = np.linspace(7900, 8000, 100).astype(int)
for count, x in enumerate(frames):
    path_main = os.path.join(path, "Q_{}.png".format(x))
    img_main = cv2.imread(path_main)
    shape = img_main.shape

    # Find Contours in Images
    img_main, vortex_centers = vortex_contours(img_main)
    if len(vortex_centers) > 0:
        red_row = [count, int(0)]
        for i in vortex_centers:
            red_1 = i[0]
            red_row.append(red_1)
            red_2 = i[1]
            red_row.append(red_2)
    else:
        red_row = []

    with open(file_csv, "a", newline='') as i:
        writer_func = csv.writer(i, delimiter=",", skipinitialspace=False)
        writer_func.writerow(red_row)
        time.sleep(1)
        i.close()

    h = len(img_main[:, 1])
    w = len(img_main[1, :])
    pt1 = (v_line, 0)
    pt2 = (v_line, h)
    pt3 = (0, h_line)
    pt4 = (w, h_line)

    cv2.line(img_main, pt1, pt2, (0, 255, 0), 1)
    cv2.line(img_main, pt3, pt4, (255, 0, 0), 1)
    cv2.imshow('Img Red {}'.format(x), img_main)
    cv2.imwrite(save_path + "/_{}_Vortex_Main.tiff".format(count), img_main)

# Finish Program
cv2.waitKey()
cv2.destroyAllWindows()
