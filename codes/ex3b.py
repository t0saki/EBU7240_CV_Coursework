# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/building.jpg')
# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

# Convert image to grayscale
gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 40, 100, apertureSize=3)

# Apply Probabilistic Hough Line Transform
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength, maxLineGap)

# Draw detected line segments on the original image
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(im1, (x1, y1), (x2, y2), (0, 0, 255), 2)

im_result = im1

##########################################################################################

cv2.imwrite('../results/ex3b_building_hough.jpg', im_result)
