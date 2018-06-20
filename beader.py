from vidstab import VidStab
import cv2
import numpy as np
import argparse
import imutils
from imutils.perspective import four_point_transform

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

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

frameData = {}

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

    custom = cv2.GaussianBlur(img, (5, 5), 0)
    custom = cv2.addWeighted(img, 1.5, custom, -0.5, 0)

#    cv2.imshow("mask", imutils.resize(img, height=300))
#    cv2.waitKey()
#    cv2.imshow("mask", imutils.resize(custom, height=300))
#    cv2.waitKey()

    img = custom

    circles = cv2.HoughCircles(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 2, 30, param1=80)
    #print(circles)
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    maxRadius = 65

    trueCircles = []

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        if i[2] <= maxRadius:
            if (i[0] > 450 and i[0] < 500) or (i[0] < 1050 and i[0] > 1000):
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),5)
                trueCircles.append(((i[0], i[1]), i[2]))
                # draw the center of the circle
                cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
            else:
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,255),3)

    #cv2.imshow("mask", imutils.resize(img, height=300))
    #cv2.waitKey()

    if len(trueCircles) != 4:
        frameData[index] = None
    else:
        centers = []
        for circle in trueCircles:
            centers.append(circle[0])

        img = four_point_transform(oimg, np.asarray(centers))
        thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        thresh = cv2.erode(thresh, kernel, iterations=4)
        thresh = cv2.dilate(thresh, kernel, iterations=12)

        ret, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chosenContour = contours[0]
        for contour in contours:
            if cv2.contourArea(contour) > cv2.contourArea(chosenContour):
                chosenContour = contour

        cv2.drawContours(img, [chosenContour], -1, (0, 255, 0), 2)
        frameData[index] = cv2.boundingRect(chosenContour)[2]
        x, y, w, h = cv2.boundingRect(chosenContour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("mask", img)
        cv2.waitKey()

    print(" ", index, end='\r')
    index += 1

#print()
#print("{")
#for i in range(0, max(frameData.keys())):
#    if not i in frameData.keys():
#        continue
#    if frameData[i] != None:
#        print("  {%d, %d}," % (i, frameData[i]))
#print("}")



