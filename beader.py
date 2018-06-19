from vidstab import VidStab
import cv2
import numpy as np
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-f", required=True, help="video path")
args = vars(ap.parse_args())
fullFilePath = args["f"]

fileName = fullFilePath.split('/')[-1]
filePath = '/'.join(fullFilePath.split('/')[:-1]) + '/'

#stabilizer = VidStab(kp_method="FAST")
#stabilizer.stabilize(input_path=fullFilePath, output_path=filePath+"out.mov")

video = cv2.VideoCapture(fullFilePath)

index = 1
img = None

def get_next_frame():
    global img
    succ, img = video.read()
    return succ

while get_next_frame():

    lowerWhite = np.array([200, 200, 200])
    upperWhite = np.array([255, 255, 255])

    mask = cv2.inRange(img, lowerWhite, upperWhite)

    thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    ret, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    copy = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(copy, contours, -1, (0, 255, 0), 5)
    #cv2.imshow("mask", imutils.resize(copy, height=300))
    #cv2.waitKey()

    shape = img.shape[:2]
#    print(shape[0]*shape[1])

    chosenContour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(chosenContour) and cv2.contourArea(contour) < (shape[0] * shape[1] * 0.4):
            chosenContour = contour

    edgeBounds = cv2.boundingRect(chosenContour)
#    print(cv2.contourArea(chosenContour))
    #cv2.rectangle(copy, edgeBounds[:2], (edgeBounds[0]+edgeBounds[2], edgeBounds[1]+edgeBounds[3]), (0, 0, 255), 5)

    bound = edgeBounds[0] + edgeBounds[2]

    height, width = img.shape[:2]
    img = img[0:height, bound:width]
    oimg = img

    """
    img = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    img = 255-img
    img = cv2.erode(img, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(img, kernel, iterations=10)
    img = cv2.erode(img, kernel, iterations=9)
    """

    """
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    """

    #cv2.imshow("mask", imutils.resize(img, height=300))
    #cv2.waitKey()

    circles = cv2.HoughCircles(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 2, 50)
    #print(circles)
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),5)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow("mask", imutils.resize(img, height=300))

    cv2.waitKey()


