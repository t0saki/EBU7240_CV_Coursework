# Object tracking with your image

import numpy as np
import cv2
from imwriter import ImageWriter


cap = cv2.VideoCapture('../inputs/ebu7240_hand_2.mp4')

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

iw_paper = ImageWriter('../results/ex4a_paper_results', [1, 20, 40, 60, 90])
iw_id = ImageWriter('../results/ex4a_id_results', [1, 20, 40, 60, 90])

ret, frame = cap.read()

r_paper, h_paper, c_paper, w_paper = 30, 160, 60, 240
r_id, h_id, c_id, w_id = 90, 110, 120, 140
track_window_paper = (c_paper, r_paper, w_paper, h_paper)
track_window_id = (c_id, r_id, w_id, h_id)

roi_paper = frame[r_paper:r_paper+h_paper, c_paper:c_paper+w_paper]
roi_id = frame[r_id:r_id+h_id, c_id:c_id+w_id]

hsv_roi_paper = cv2.cvtColor(roi_paper, cv2.COLOR_BGR2HSV)
hsv_roi_id = cv2.cvtColor(roi_id, cv2.COLOR_BGR2HSV)

mask_paper = cv2.inRange(hsv_roi_paper, np.array(
    (0., 60., 32.)), np.array((180., 255., 255.)))
mask_id = cv2.inRange(hsv_roi_id, np.array(
    (0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist_paper = cv2.calcHist(
    [hsv_roi_paper], [0], mask_paper, [180], [0, 180])
roi_hist_id = cv2.calcHist([hsv_roi_id], [0], mask_id, [180], [0, 180])

cv2.normalize(roi_hist_paper, roi_hist_paper, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(roi_hist_id, roi_hist_id, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def meanshift_tracker(cap, vw, name, output_path, window):
    # Initialize the tracking window
    r, h, c, w = window
    track_window = (c, r, w, h)

    # Read the first frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Unable to read video")
        return

    # Set the region of interest for the object to track
    roi = frame[r:r+h, c:c+w]

    # Convert the roi to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create a mask for the histogram
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))

    # Compute the histogram for the roi
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    # Normalize the histogram
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria: either 10 iterations or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw the tracking window on the image
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        vw.write(frame)

    vw.release()
    cap.release()
    cv2.destroyAllWindows()

##########################################################################################


out_paper = cv2.VideoWriter(
    '../results/ex4a_meanshift_track_paper.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
out_id = cv2.VideoWriter(
    '../results/ex4a_meanshift_track_id.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst_paper = cv2.calcBackProject(
        [hsv.copy()], [0], roi_hist_paper, [0, 180], 1)
    dst_id = cv2.calcBackProject([hsv.copy()], [0], roi_hist_id, [0, 180], 1)

    frame_paper = frame.copy()
    frame_id = frame.copy()

    ret, track_window_paper = cv2.meanShift(
        dst_paper, track_window_paper, term_crit)
    x_paper, y_paper, w_paper, h_paper = track_window_paper
    cv2.rectangle(frame_paper, (x_paper, y_paper),
                  (x_paper+w_paper, y_paper+h_paper), 255, 2)
    out_paper.write(frame_paper)

    ret, track_window_id = cv2.meanShift(dst_id, track_window_id, term_crit)
    x_id, y_id, w_id, h_id = track_window_id
    cv2.rectangle(frame_id, (x_id, y_id), (x_id+w_id, y_id+h_id), 255, 2)
    out_id.write(frame_id)

cap.release()
out_paper.release()
out_id.release()
cv2.destroyAllWindows()
