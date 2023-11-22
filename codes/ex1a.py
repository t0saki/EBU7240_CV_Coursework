import cv2

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

import os

frame_count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        frame = cv2.resize(frame, (640, 360))
        
        if 1 <= frame_count <= 30: # Full color
            pass
        elif 31 <= frame_count <= 50: # Zero values to R, B channel
            frame[:,:,2] = 0 # Red channel
            frame[:,:,0] = 0 # Blue channel
        elif 51 <= frame_count <= 70: # Zero values to B, G channel
            frame[:,:,0] = 0 # Blue channel
            frame[:,:,1] = 0 # Green channel
        elif 71 <= frame_count <= 90: # Zero values to R, G channel
            frame[:,:,2] = 0 # Red channel
            frame[:,:,1] = 0 # Green channel

        img_array.append(frame)

        # Save specified frames as images
        os.makedirs('../results/ex1_a_results', exist_ok=True)
        if frame_count in [1, 21, 31, 61, 90]:
            cv2.imwrite(f'../results/ex1_a_results/{frame_count}.png', frame)
    else:
        break

cap.release()

size = (640, 360)

##########################################################################################

out = cv2.VideoWriter('../results/ex1_a_hand_rgbtest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
