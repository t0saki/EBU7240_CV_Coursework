import cv2


cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

im_myname = cv2.imread('../inputs/my_name.png')

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

import os

frame_count = 0
ex_name = 'ex1_b'

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        if frame_count <= 90:
            # Define the top left corner of the image
            start_row, start_col = frame.shape[0] - im_myname.shape[0], frame_count * 2

            # Overlay the image on the frame
            frame[start_row:start_row + im_myname.shape[0], start_col:start_col + im_myname.shape[1]] = im_myname

            # Save specified frames as images
            os.makedirs(f'../results/{ex_name}_results', exist_ok=True)
            if frame_count in [1, 21, 31, 61, 90]:
                cv2.imwrite(f'../results/{ex_name}_results/{frame_count}.png', frame)

        img_array.append(frame)
    else:
        break

cap.release()

size = (640, 360)

##########################################################################################


out = cv2.VideoWriter('../results/ex1_b_hand_composition.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
