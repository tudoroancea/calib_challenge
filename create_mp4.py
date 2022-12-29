import numpy as np
import torch
import cv2

for i in range(5,10):
    # open video from labeled dir
    cap = cv2.VideoCapture(f'unlabeled/{i}.hevc')
    # convert video to mp4 with right height and width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    out = cv2.VideoWriter(f'unlabeled/{i}.mp4', fourcc, 20.0, (int(width), int(height)))
    # read video frame by frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # write the frame
            out.write(frame)
        else:
            break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
