# Bilateral filtering without OpenCV
import numpy as np
import cv2
import sys
import math


#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE

def bilateral_filter_gray(im_gray, window_size, sigma_r, sigma_s):
    half_size = window_size // 2
    im_filtered = np.zeros_like(im_gray, dtype=np.float64)
    im_padded = np.pad(im_gray, half_size, mode='reflect')

    for i in range(half_size, im_padded.shape[0] - half_size):
        for j in range(half_size, im_padded.shape[1] - half_size):
            region = im_padded[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            intensity_diff = region - im_padded[i, j]
            spatial_diff = np.sqrt((np.arange(-half_size, half_size+1)[:, None])**2 + (np.arange(-half_size, half_size+1)**2))

            weights = np.exp(-(intensity_diff**2)/(2*sigma_r**2)) * np.exp(-(spatial_diff**2)/(2*sigma_s**2))
            weights /= np.sum(weights)
            im_filtered[i-half_size, j-half_size] = np.sum(region * weights)

    return im_filtered

##########################################################################################


im_gray = cv2.imread('../inputs/cat.png',0)

result_bf1 = bilateral_filter_gray(im_gray, 11, 30.0, 3.0)
result_bf2 = bilateral_filter_gray(im_gray, 11, 30.0, 30.0)
result_bf3 = bilateral_filter_gray(im_gray, 11, 100.0, 3.0)
result_bf4 = bilateral_filter_gray(im_gray, 11, 100.0, 30.0)
result_bf5 = bilateral_filter_gray(im_gray, 5, 100.0, 30.0)

result_bf1 = np.uint8(result_bf1)
result_bf2 = np.uint8(result_bf2)
result_bf3 = np.uint8(result_bf3)
result_bf4 = np.uint8(result_bf4)
result_bf5 = np.uint8(result_bf5)


cv2.imwrite('../results/ex2b_bf_11_30_3.jpg', result_bf1)
cv2.imwrite('../results/ex2b_bf_11_30_30.jpg', result_bf2)
cv2.imwrite('../results/ex2b_bf_11_100_3.jpg', result_bf3)
cv2.imwrite('../results/ex2b_bf_11_100_30.jpg', result_bf4)
cv2.imwrite('../results/ex2b_bf_5_100_30.jpg', result_bf5)

