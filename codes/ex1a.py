import os
import cv2
from imwriter import ImageWriter

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#


frame_count = 0

iw = ImageWriter('../results/ex1_a_results', [1, 21, 31, 61, 90])

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if 1 <= frame_count <= 30:
            pass
        elif 31 <= frame_count <= 50:
            frame[:, :, 2] = 0
            frame[:, :, 0] = 0
        elif 51 <= frame_count <= 70:
            frame[:, :, 0] = 0
            frame[:, :, 1] = 0
        elif 71 <= frame_count <= 90:
            frame[:, :, 2] = 0
            frame[:, :, 1] = 0

        img_array.append(frame)

        iw.write(frame)
    else:
        break

cap.release()

size = (360, 640)

##########################################################################################

out = cv2.VideoWriter('../results/ex1_a_hand_rgbtest.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
