import os
import cv2
from .imwriter import ImageWriter

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

im_myname = cv2.imread('../inputs/my_name.png')

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

iw = ImageWriter('../results/ex1_b_results', [1, 21, 31, 61, 90])

frame_count = 0
ex_name = 'ex1_b'

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        if frame_count <= 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            start_row, start_col = frame.shape[0] - \
                im_myname.shape[0], frame_count * 2

            print(
                f'Frame {frame_count}: start_row={start_row}, start_col={start_col}')

            frame[start_row:start_row + im_myname.shape[0],
                  start_col:start_col + im_myname.shape[1]] = im_myname

            iw.write(frame)

        img_array.append(frame)
    else:
        break

cap.release()

size = (360, 640)

##########################################################################################


out = cv2.VideoWriter('../results/ex1_b_hand_composition.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
