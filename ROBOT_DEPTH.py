#################################################################
###
### Program to compute depth from stereo camera
### Uses open cv
###
#################################################################


import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


cap1 = cv2.VideoCapture(0);
cap2 = cv2.VideoCapture(1);
# imgL = cv2.imread('images/l_94_3.png', cv2.IMREAD_GRAYSCALE)
# imgR = cv2.imread('images/r_94_3.png', cv2.IMREAD_GRAYSCALE)

fig=plt.figure()
plt.ion()
plt.show(block=False)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, imgL = cap1.read()
    ret, imgR = cap2.read()

    # imgL = cv2.resize(imgL, (0,0), fx=0.8, fy=0.8)
    # imgR = cv2.resize(imgR, (0,0), fx=0.8, fy=0.8)

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    ret,imgL = cv2.threshold(imgL, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,imgR = cv2.threshold(imgR, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp

    stereo = cv2.StereoSGBM(minDisparity = min_disp,
                    numDisparities = num_disp,
                    SADWindowSize = window_size,
                    uniquenessRatio = 10,
                    speckleWindowSize = 100,
                    speckleRange = 32,
                    disp12MaxDiff = 1,
                    P1 = 8*3*window_size**2,
                    P2 = 32*3*window_size**2,
                    fullDP = False
            )

    # stereo = cv2.StereoSGBM(0, 32, 3, 128, 256, 20, 16, 1, 100, 20, True)
    disparity = stereo.compute(imgL,imgR)

    # plt.figure()
    plt.subplot(221), plt.imshow(imgL, 'gray')
    plt.subplot(222), plt.imshow(imgR, 'gray')
    plt.subplot(223), plt.imshow(disparity)
    plt.draw()


cap2.release()
cap1.release()
cv2.destroyAllWindows()
