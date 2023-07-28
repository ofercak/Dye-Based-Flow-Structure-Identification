# Vortex Tracking Algorithm [Dye Visualization Parameters]
# Ondrej Fercak, 7/22/2023

import cv2
import numpy as np
import csv
import time
import glob
import os

area_thresh = 0.3
area_min = 1600
area_max = 6000

iterations = 4
element = 5
blur = 7

scale = 0.4


def vortex_init(img, scl):
    height = shape[0]
    width = shape[1]
    if scl != 1:
        img = cv2.resize(img_main, (int(scl * width), int(scl * height)), interpolation=cv2.INTER_AREA)

    # Create a white image, a window
    canvas_white = np.ones((int(scl * height), int(scl * width), 3), np.uint8) * 255
    canvas_black = np.ones((int(scl * height), int(scl * width), 3), np.uint8)

    return img, canvas_white, canvas_black


def erode_dilate(img, canvas_white, canvas_black):
    gray = np.where(img < 255, canvas_white, canvas_black)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (element, element))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, elem, iterations=iterations)
    gray = cv2.medianBlur(gray, blur)
    gray = cv2.dilate(gray, elem, iterations=1)
    return gray


def vortex_contours(img, img_color, gray, canvas_white, color):
    if color == 'r' or color == 'red':
        disp_color = (70, 70, 255)
    elif color == 'b' or color == 'blue':
        disp_color = (255, 70, 70)
    else:
        disp_color = (0, 255, 0)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    vortex_centers = []
    img_2 = cv2.cvtColor(img_color, cv2.COLOR_GRAY2RGB)

    # list for storing names of shapes
    for cnt in contours:
        blob_area = cv2.contourArea(cnt)

        if area_min < blob_area < area_max:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circle_area = int(np.pi*radius**2)

            if blob_area / circle_area > area_thresh:
                print("Blob Area = {}".format(blob_area))
                vortex_centers.append((int(cx), int(cy)))

                cv2.circle(img, (int(cx), int(cy)), int(radius), disp_color, 5)
                cv2.circle(img_2, (int(cx), int(cy)), int(radius), disp_color, 5)

    return img, img_2, np.array(vortex_centers)


def sort_distance(coordinates):
    if len(coordinates) >= 2:
        dist = np.zeros((len(coordinates), 1))
        result = np.zeros((len(coordinates), 5))

        for i, center in enumerate(coordinates[:, 0]):
            center_x = coordinates[i, 0]
            center_y = coordinates[i, 1]
            result[i, 0] = center_x
            result[i, 1] = center_y

            for j, comp in enumerate(coordinates):
                comp_x = coordinates[j, 0]
                comp_y = coordinates[j, 1]
                dist[j] = np.sqrt((center_x - comp_x) ** 2 + (center_y - comp_y) ** 2)

            dist_sort = np.sort(dist, axis=0)
            match = np.where(dist[:, 0] == (dist_sort[1, 0]))
            result[i, 2] = coordinates[match, 0]
            result[i, 3] = coordinates[match, 1]
            result[i, 4] = dist_sort[1, 0] * 1/137
    else:
        result = []

    return result


def vortex_lengths(image, centers):
    avg_lengths = []
    if centers is None:
        image = image
        avg_lengths.append(np.nan)
    else:
        for k, line in enumerate(centers):
            sx = int(line[0])
            sy = int(line[1])
            ex = int(line[2])
            ey = int(line[3])
            vd = round((line[4]), 3)
            avg_lengths.append(vd)
            cv2.line(image, (sx, sy), (ex, ey), (250, 100, 0), 3)
            cv2.putText(image, '{}"'.format(vd), (int((ex + sx)/2), int((ey + sy)/2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (250, 100, 0), 3, cv2.LINE_AA)

    print("Average Lengths = {}".format(np.mean(avg_lengths)))

    return image


####################################################################################################
# MAIN
####################################################################################################

path = "***Your Path Here***"
save_path = "***Your Path Here***"

file_csv = path + "/Vortex Distances.csv"
header = ["Image #", "Color", "X-Y Locations"]

with open(file_csv, 'x', newline='') as f:
    writer = csv.writer(f, delimiter=",", skipinitialspace=True)
    writer.writerow(header)
    time.sleep(1)
    f.close()

frames = np.linspace(1, 200, 200).astype(int)
for count, x in enumerate(frames):
    path_main = os.path.join(path, "Dye_Vis_{}.jpg".format(x))
    path_red = os.path.join(path, "Dye_Vis_{}.jpg_Red.tiff".format(x))
    path_blue = os.path.join(path, "Dye_Vis_{}.jpg_Blue.tiff".format(x))

    img_main = cv2.imread(path_main)
    img_red = cv2.imread(path_red)
    img_blue = cv2.imread(path_blue)

    shape = img_red.shape

    # Initialize Images
    img_main, canvas_white_main, canvas_black_main = vortex_init(img_main, 1)
    img_red, canvas_white_red, canvas_black_red = vortex_init(img_red, 1)
    img_blue, canvas_white_blue, canvas_black_blue = vortex_init(img_blue, 1)

    # Apply Smoothing & Filtering to Images
    gray_red = erode_dilate(img_red, canvas_white_red, canvas_black_red)
    gray_blue = erode_dilate(img_blue, canvas_white_blue, canvas_black_blue)
    gray_red_show = cv2.resize(gray_red, (int(scale * shape[1]), int(scale * shape[0])), interpolation=cv2.INTER_AREA)
    gray_blue_show = cv2.resize(gray_blue, (int(scale * shape[1]), int(scale * shape[0])), interpolation=cv2.INTER_AREA)
    cv2.imshow('Gray Red', gray_red_show)
    cv2.imshow('Gray Blue', gray_blue_show)
    cv2.imwrite(save_path + "/_{}_Vortex_Red_Mask.tiff".format(count), gray_red)

    # Find Contours in Images
    img_main, img_red, vortex_centers_red = vortex_contours(img_main, gray_red, gray_red, canvas_white_red, 'r')
    img_main, img_blue, vortex_centers_blue = vortex_contours(img_main, gray_blue, gray_blue, canvas_white_blue, 'b')
    cv2.imshow("Here", img_red)
    if len(vortex_centers_red) > 0:
        red_row = [count, int(0)]
        for i in vortex_centers_red:
            red_1 = i[0]
            red_row.append(red_1)
            red_2 = i[1]
            red_row.append(red_2)
    else:
        red_row = []

    if len(vortex_centers_blue) > 0:
        blue_row = [count, int(1)]
        for i in vortex_centers_blue:
            blue_1 = i[0]
            blue_row.append(blue_1)
            blue_2 = i[1]
            blue_row.append(blue_2)
    else:
        blue_row = []

    with open(file_csv, "a", newline='') as i:
        writer_func = csv.writer(i, delimiter=",", skipinitialspace=False)
        writer_func.writerow(red_row)
        writer_func.writerow(blue_row)
        time.sleep(1)
        i.close()

    img_red_show = cv2.resize(img_red, (int(scale * shape[1]), int(scale * shape[0])), interpolation=cv2.INTER_AREA)
    img_blue_show = cv2.resize(img_blue, (int(scale * shape[1]), int(scale * shape[0])), interpolation=cv2.INTER_AREA)
    cv2.imshow('Img Red', img_red_show)
    cv2.imshow('Img Blue', img_blue_show)

    # Find Best Fit Distances Between Vortex Centers
    red_centers = sort_distance(vortex_centers_red)
    blue_centers = sort_distance(vortex_centers_blue)

    # Display Best Fit Distances Between Vortex Centers
    img_main = vortex_lengths(img_main, red_centers)
    img_main = vortex_lengths(img_main, blue_centers)
    img_red = vortex_lengths(img_red, red_centers)
    img_blue = vortex_lengths(img_blue, blue_centers)

    cv2.imwrite(save_path + "/_{}_Vortex_Red.tiff".format(count), img_red)
    cv2.imwrite(save_path + "/_{}_Vortex_Blue.tiff".format(count), img_blue)
    cv2.imwrite(save_path + "/_{}_Vortex_Main.tiff".format(count), img_main)

    # Visualize Images
    final = cv2.hconcat((img_main, img_red))
    final = cv2.hconcat((final, img_blue))

    final = cv2.resize(final, (int(3 * scale * shape[1]), int(scale * shape[0])), interpolation=cv2.INTER_AREA)
    cv2.imshow('Vortex Identification', final)
    cv2.imwrite(save_path + "/_{}_Vortex.tiff".format(count), final)

# Finish Program
cv2.waitKey()
cv2.destroyAllWindows()
