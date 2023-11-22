import numpy as np
import cv2
import math


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def matching_cost_computation(matching_cost, left_img, right_img, d_max):
    # Matching cost computation: SSD
    h, w = left_img.shape
    for y in range(h):
        for x in range(w):
            for d in range(d_max):
                if x - d >= 0:
                    temp_cost = float(left_img[y, x]) - \
                        float(right_img[y, x - d])
                    matching_cost[y, x, d] = temp_cost * temp_cost


def cost_aggregation_adaptiveweight(aggregated_cost, matching_cost, left_img, kernel, sigma_r, sigma_s, d_max):
    # --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
    # NO OPENCV FUNCTION IS ALLOWED HERE

    h, w = left_img.shape
    r = kernel_size // 2  # radius of the window

    # Iterate over each pixel
    for y in range(h):
        for x in range(w):
            for d in range(d_max):

                # Define the window boundaries
                y_min, y_max = max(y - r, 0), min(y + r, h - 1)
                x_min, x_max = max(x - r, 0), min(x + r, w - 1)

                # Initialize the sums used to compute the weighted average
                weight_sum, cost_sum = 0.0, 0.0

                # Iterate over each pixel in the window
                for j in range(y_min, y_max + 1):
                    for i in range(x_min, x_max + 1):

                        # Compute the intensity and spatial distances
                        intensity_distance = np.abs(
                            int(left_img[j, i]) - int(left_img[y, x]))
                        spatial_distance = distance(x, y, i, j)

                        # Compute the weights using the Gaussian functions
                        weight = gaussian(
                            intensity_distance, sigma_r) * gaussian(spatial_distance, sigma_s)

                        # Update the sums
                        weight_sum += weight
                        cost_sum += weight * matching_cost[j, i, d]

                # Compute the weighted average
                aggregated_cost[y, x, d] = cost_sum / \
                    weight_sum if weight_sum > 0 else 0

##########################################################################################


def disparity_estimation(disparity, aggregated_cost, d_max):
    offset_adjust = 255 / d_max  # this is used to map disparity map output to 0-255 range
    h, w = disparity.shape
    for y in range(h):
        for x in range(w):
            best_offset = 0
            prev_ssd = 65534
            for d in range(d_max):
                if aggregated_cost[y, x, d] < prev_ssd:
                    prev_ssd = aggregated_cost[y, x, d]
                    best_offset = d

            # set disparity output for this x,y location to the best match
            disparity[y, x] = best_offset * offset_adjust


if __name__ == '__main__':
    # Load left and right images and convert to grayscale for simplicity
    left_img = cv2.imread('../inputs/teddy_im2.png', 0)
    right_img = cv2.imread('../inputs/teddy_im6.png', 0)

    h, w = left_img.shape
    d_max, kernel_size = 32, 3
    # parameters used for bilateral filter's intensity and spatial distance
    sigma_r, sigma_s = 100.0, 30.0

    matching_cost = np.zeros((h, w, d_max), np.float64)
    aggregated_cost = np.zeros((h, w, d_max), np.float64)
    disparity = np.zeros((h, w), np.uint8)

    matching_cost_computation(matching_cost, left_img, right_img, d_max)
    # COMPLETE THIS FUNCTION
    cost_aggregation_adaptiveweight(
        aggregated_cost, matching_cost, left_img, kernel_size, sigma_r, sigma_s, d_max)
    ##
    disparity_estimation(disparity, aggregated_cost, d_max)
    cv2.imwrite('../results/ex3d_aw_3.png', disparity)
    print('done1')

    d_max, kernel_size = 32, 11

    matching_cost = np.zeros((h, w, d_max), np.float64)
    aggregated_cost = np.zeros((h, w, d_max), np.float64)
    disparity = np.zeros((h, w), np.uint8)

    matching_cost_computation(matching_cost, left_img, right_img, d_max)
    # COMPLETE THIS FUNCTION
    cost_aggregation_adaptiveweight(
        aggregated_cost, matching_cost, left_img, kernel_size, sigma_r, sigma_s, d_max)
    ##
    disparity_estimation(disparity, aggregated_cost, d_max)
    cv2.imwrite('../results/ex3d_aw_11.png', disparity)
    print('done2')
