from vidstab import VidStab
import cv2
import numpy as np
import argparse
import imutils
from math import sqrt
from imutils.perspective import four_point_transform
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

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

def distance_between(point1, point2):
    return sqrt(abs(point1[0]-point2[0])**2+abs(point1[1]-point2[1])**2)

frameData = {}

while get_next_frame():

    oimg = img.copy()

    lowerRed = np.array([0, 150, 150])
    upperRed = np.array([5, 255, 255])

    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowerRed, upperRed)

    lowerRed = np.array([170, 120, 120])
    upperRed = np.array([180, 255, 255])

    #we need two red ranges because red hue is split in hsv

    mask += cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowerRed, upperRed)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations = 20)

    ret, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #copy = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(copy, contours, -1, (0, 255, 0), 1)
    #cv2.imshow("copy", copy)
    #cv2.waitKey()
    centers = []
    contures = []
    for contour in contours:
        circle = cv2.minEnclosingCircle(contour)
        for i, conture in enumerate(contures):
            circul = cv2.minEnclosingCircle(conture)
            if distance_between(circul[0], circle[0]) < 200:
                combinedContour = np.vstack([contour, conture])
                combinedContour = cv2.convexHull(combinedContour)
                contures[i] = combinedContour
                break
        else:
            contures.append(contour)
    for conture in contures:
        centers.append(cv2.minEnclosingCircle(conture)[0])


    if len(centers) != 4:
        frameData[index] = None
        print("could not locate at frame %d" % index)
    else:
        img = four_point_transform(oimg, np.asarray(centers))
        simg = cv2.pyrMeanShiftFiltering(img, 21, 51)
        #simg = img.copy()
        thresh = cv2.threshold(cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        thresh = cv2.erode(thresh, kernel, iterations=3)
        thresh = cv2.dilate(thresh, kernel, iterations=9)

        ret, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #chosenContour = contours[0]
        chosenContours = []
        for contour in contours:
            #if cv2.contourArea(contour) > cv2.contourArea(chosenContour):
            rect = cv2.boundingRect(contour)
            shape = img.shape[:2]
            if rect[0]>shape[1]*0.2 and rect[1]>shape[0]*0.2 and rect[0]+rect[2]<shape[1]*0.8 and rect[1]+rect[3]<shape[0]*0.8:
                #just making sure that its in the middle of the frame
                chosenContours.append(contour)

        chosenContour = np.vstack(chosenContours)

        oimg = img.copy()
        cv2.drawContours(img, chosenContours, -1, (0, 255, 0), 2)
        # length, width, area, area/length, area/width
        x, y, w, h = cv2.boundingRect(chosenContour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        a = 0
        for contour in chosenContours:
            a += cv2.contourArea(contour)
        frameData[index] = {'l': w, 'w': h, 'a': a, 'al': a/w, 'aw': a/h}

        cv2.imshow("img", cv2.resize(img, (440, 300)))

        """
        img = oimg
        shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
        #cv2.imshow("mask", cv2.resize(shifted, (440, 300)))
        #cv2.waitKey()

        lowerBrown = np.array([2, 30, 0])
        upperBrown = np.array([16, 180, 200])

        thresh1 = cv2.inRange(cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV), lowerBrown, upperBrown)
        #cv2.imshow("mask", cv2.resize(thresh1, (440, 300)))
        #cv2.waitKey()

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #for x in range(1, 6):
        #    thresh2 = cv2.threshold(grey, int(255 * x * 0.1), 255, cv2.THRESH_BINARY_INV)[1]
            #cv2.imshow("mask", cv2.resize(thresh2, (440, 300)))
            #cv2.waitKey()

        #    thresh3 = thresh1 & thresh2

        #    cv2.imshow("mask%d" % x, cv2.resize(thresh3, (440, 300)))
        thresh4 = thresh1 & (255-grey)
        cv2.imshow("maskPro", cv2.resize(thresh4, (440, 300)))
        """
        cv2.waitKey()

    index += 1

#metric output stuff
for i in range(0, max(frameData.keys())):
    if not i in frameData.keys():
        continue
    if frameData[i] != None:
        for k in frameData[i].keys():
            outFileName = "data_" + k
            outFullFilePath = filePath + outFileName
            outFile = open(outFullFilePath, 'a')
            outFile.write("{%d, %f},\n" % (i, frameData[i][k]))
            outFile.close()




