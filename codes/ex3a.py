# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/Img01.jpg')
im2 = cv2.imread('../inputs/Img02.jpg')


im_gray1 = cv2.imread('../inputs/Img01.jpg', 0)
im_gray2 = cv2.imread('../inputs/Img02.jpg', 0)

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

# Create SIFT object
sift = cv2.SIFT_create()

# Compute SIFT descriptors and keypoints
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract location of good matches
points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Find Affine Transform without RANSAC
M_noRANSAC = cv2.estimateAffinePartial2D(points1, points2)[0]

# Apply RANSAC
M_RANSAC = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)[0]


def remove_black_edges(image):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    rows = np.any(flatImage, axis=1)
    cols = np.any(flatImage, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return image[rmin:rmax+1, cmin:cmax+1]

# Function to stitch images using given transformation matrix


def stitch_image(im1, im2, M):
    # Calculate size of new image
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.transform(pts2, M)
    pts = np.concatenate((pts1, pts2_), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel())
    t = [-xmin, -ymin]

    # Append translation to the transformation matrix
    M[0, 2] += t[0]
    M[1, 2] += t[1]

    result = np.zeros(
        (max(ymax-ymin, h2+t[1]), max(xmax-xmin, w2+t[0]), 3), dtype=np.uint8)

    # Warp image 1 to align it to image 2 and stitch them together
    result[:ymax-ymin, :xmax -
           xmin] = cv2.warpAffine(im1, M, (xmax-xmin, ymax-ymin))
    result[t[1]:h2+t[1], t[0]:w2+t[0]] = im2

    # Remove black portion of the image
    result = remove_black_edges(result)

    return result


# Perform image stitching
panorama_noRANSAC = stitch_image(im1, im2, M_noRANSAC)
panorama_RANSAC = stitch_image(im1, im2, M_RANSAC)

##########################################################################################

cv2.imwrite('../results/ex3a_stitched_noRANSAC.jpg', panorama_noRANSAC)
cv2.imwrite('../results/ex3a_stitched_RANSAC.jpg', panorama_RANSAC)
