import cv2

import numpy as np
 
capture = cv2.VideoCapture("video.mp4")
count=0
history = 500
nGauss = 6
bgThresh = 0.6
noise = 1 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history,nGauss,bgThresh,noise)
 

while(1):
    ret, frame = capture.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
 

print('Done!')

