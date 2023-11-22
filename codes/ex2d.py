# SIFT matching using OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt


im_gray1 = cv2.imread('../inputs/sift_input1.jpg', 0)
im_gray2 = cv2.imread('../inputs/sift_input2.jpg', 0)

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

# # Debug for ex3a
# im_gray1 = cv2.imread('../inputs/Img01.jpg', 0)
# im_gray2 = cv2.imread('../inputs/Img02.jpg', 0)

# Create SIFT object
sift = cv2.SIFT_create()

# Compute SIFT descriptors and keypoints
kp1, des1 = sift.detectAndCompute(im_gray1, None)
kp2, des2 = sift.detectAndCompute(im_gray2, None)

# Draw keypoints on the images
img_sift_kp_1 = cv2.drawKeypoints(im_gray1, kp1, None)
img_sift_kp_2 = cv2.drawKeypoints(im_gray2, kp2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the best 10 and worst 10 matches.
img_most10 = cv2.drawMatches(im_gray1, kp1, im_gray2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_least10 = cv2.drawMatches(im_gray1, kp1, im_gray2, kp2, matches[-10:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Bug of this question?
img_least50 = img_least10
img_most50 = img_most10

##########################################################################################

# Keypoint maps
cv2.imwrite('../results/ex2d_sift_input1.jpg', np.uint8(img_sift_kp_1))
cv2.imwrite('../results/ex2d_sift_input2.jpg', np.uint8(img_sift_kp_2))


# Feature Matching outputs
cv2.imwrite('../results/ex2d_matches_least10.jpg', np.uint8(img_least50))
cv2.imwrite('../results/ex2d_matches_most10.jpg', np.uint8(img_most50))