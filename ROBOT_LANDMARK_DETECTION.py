#################################################################
###
### Program to detect landmarks, real time using camera
### Landmarks detected based on arcLength of their contours
### Uses open cv
###
#################################################################

import numpy as np
import cv2
from matplotlib import pyplot as plt

# draw a circle around the contour on the image
# thisContour - contour around which the circle is drawn
# onThisImage - image on which the circle is drawn
# isContourDrawn - boolean - should contour be drawn
def drawCircleAround(thisContour, onThisImage, isContourDrawn):
    (x,y),radius = cv2.minEnclosingCircle(thisContour)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle( onThisImage,
                center,
                radius,
                (255,255,255),
                2,
                8 )

    # draw contours
    if isContourDrawn:
        cv2.drawContours(
                image=onThisImage,
                contours=cnt,
                contourIdx=-1,
                color=(255,255,255),
                thickness=-1)


    return center, radius



# get the largest contour based on arcLength
# contours - the list of contours
# return values :
#       cnt - largest contour
#       max_index - index of largest contour
#       min_index - index of smallest contour
def getLargestContour(contours):
    # mark contour with largest area
    # areas = [cv2.contourArea(c) for c in contours]
    areas = [cv2.arcLength(c, True) for c in contours]
    max_index = np.argmax(areas)
    min_index = np.argmin(areas)
    cnt=contours[max_index]
    return cnt, max_index, min_index


def markCenter(onThisImage, center):
    cv2.circle( onThisImage,
                center,
                5,
                (255,0,0),
                2,
                8 )


# each landmrk contains (center, radius)
def showOnlyLandmarks(onThisImage, landmark_list):
    onThisImage = cv2.cvtColor(onThisImage, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(onThisImage.shape, np.uint8)
    for landmark in landmark_list:
        cv2.circle(mask,landmark[0],landmark[1],(255,255,255),-2)
    masked_img = cv2.bitwise_and(img,img,mask = mask)
    return masked_img


cap = cv2.VideoCapture(0)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture frame-by-frame
    ret, img = cap.read()

    # apply mask to reduce window size
    # mask = np.zeros(img.shape[:2], np.uint8)
    # mask[100:400, 200:600] = 255
    # masked_img = cv2.bitwise_and(img,img,mask = mask)

    # vary threashold here to control the landmarks detected
    edges = cv2.Canny(img, 200, 250)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    landmark_list = []
    if contours.__len__ > 0:
        for i in range(0,4):
            cnt, max_index, min_index = getLargestContour(contours)
            center, radius = drawCircleAround(cnt, img, False)
            markCenter(img, center)
            contours[max_index] = contours[min_index]
            landmark_list.append((center, radius))

        # draw contours
        # i = 0
        # for c in contours:
        #  cv2.drawContours(img, contours,i, (255,0,0), thickness=3, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0,0))
        #  i+=1

        masked_img = showOnlyLandmarks(img,landmark_list)

        # display the image
        cv2.imshow('frame', masked_img)

cap.release()
cv2.destroyAllWindows()