# Gaussian filtering without OpenCV
import numpy as np
import cv2

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE

def gaussian_filter_gray(im_gray, window_size, std_dev):
    # Create Gaussian kernel
    kernel_size = window_size
    kernel = np.fromfunction(
        lambda x, y: (1/2*np.pi*std_dev**2)*np.exp(-((x-(kernel_size-1)/2)**2+(y-(kernel_size-1)/2)**2)/(2*std_dev**2)), 
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)

    # Apply Gaussian filter
    pad_size = kernel_size // 2
    padded_image = np.pad(im_gray, pad_size, mode='reflect').astype(float)
    im_blurred = np.zeros_like(im_gray, dtype=float)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            im_blurred[i-pad_size, j-pad_size] = np.sum(padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1] * kernel)

    return im_blurred

##########################################################################################


im_gray = cv2.imread('../inputs/lena.jpg', 0)
im_gray = cv2.resize(im_gray, (256, 256))

result_gf1 = gaussian_filter_gray(im_gray, 5, 1.0)
result_gf2 = gaussian_filter_gray(im_gray, 5, 10.0)
result_gf3 = gaussian_filter_gray(im_gray, 21, 1.0)
result_gf4 = gaussian_filter_gray(im_gray, 21, 10.0)

result_gf1 = np.uint8(result_gf1)
result_gf2 = np.uint8(result_gf2)
result_gf3 = np.uint8(result_gf3)
result_gf4 = np.uint8(result_gf4)

cv2.imwrite('../results/ex2a_gf_5_1.jpg', result_gf1)
cv2.imwrite('../results/ex2a_gf_5_10.jpg', result_gf2)
cv2.imwrite('../results/ex2a_gf_21_1.jpg', result_gf3)
cv2.imwrite('../results/ex2a_gf_21_10.jpg', result_gf4)
