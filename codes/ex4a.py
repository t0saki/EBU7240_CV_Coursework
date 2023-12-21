# Object tracking with your image

import numpy as np
import cv2
from imwriter import ImageWriter


cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

r_paper, h_paper, c_paper, w_paper = 60, 240, 170, 160
r_id, h_id, c_id, w_id = 120, 140, 160, 110


def meanshift_tracker(cap, vw, name, window):
    iw = ImageWriter(
        f'../results/ex4a_{name}_results', [1, 20, 40, 60, 90])

    r, h, c, w = window
    track_window = (c, r, w, h)

    ret, frame = cap.read()

    roi = frame[r:r+h, c:c+w]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while cap.isOpened():
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        vw.write(frame)
        iw.write(frame)

        ret, frame = cap.read()
        if not ret:
            break

    vw.release()
    cap.release()
    cv2.destroyAllWindows()

##########################################################################################


out_paper = cv2.VideoWriter(
    '../results/ex4a_meanshift_track_paper.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
out_id = cv2.VideoWriter(
    '../results/ex4a_meanshift_track_id.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

meanshift_tracker(cap, out_paper, "paper", [
                  r_paper, h_paper, c_paper, w_paper])

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')
meanshift_tracker(cap, out_id, "id", [
                  r_id, h_id, c_id, w_id])
