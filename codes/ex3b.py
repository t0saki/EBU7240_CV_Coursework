# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/building.jpg')
im_own = cv2.imread('../inputs/YOUR_OWN.jpg')
# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#


def do_HoughLinesP(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 40, 100, apertureSize=3)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120,
                            minLineLength, maxLineGap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return im

##########################################################################################


cv2.imwrite('../results/ex3b_building_hough.jpg', do_HoughLinesP(im1))
cv2.imwrite('../results/YOUR_OWN_hough.jpg', do_HoughLinesP(im_own))
