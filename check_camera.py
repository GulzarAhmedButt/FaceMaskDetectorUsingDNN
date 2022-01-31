import cv2 as cv
import numpy as np


cap = cv.VideoCapture("/dev/video3")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow("Live Camera", frame)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
