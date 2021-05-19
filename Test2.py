from edgeDetect import four_point_transform,canny_edge_detection
import numpy as np

import cv2
import imutils

path_image="120/Anh_Nghieng/OK_181001_dadae.jpg"
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(path_image)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#edged = cv2.Canny(gray, 90, 150)
edged=canny_edge_detection(path_image)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("a",edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
for k in cnts[7:9]:
	cv2.drawContours(orig, k, -1, (255, 25, 25), 5)
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		print(approx)
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = canny_edge_detection(path_image)
# warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
def paint(image,cnts):
    if cv2.contourArea(cnts) > 600:
        cv2.drawContours(image,cnts,-1,(255,25,25),5)
        peri = cv2.arcLength(cnts, True)
        approx = cv2.approxPolyDP(cnts, 0.02 * peri, True)
        left=(approx[1][0][0],approx[1][0][1])
        right=(approx[3][0][0],approx[3][0][1])
        left2 = (approx[2][0][0], approx[2][0][1])
        right2 = (approx[0][0][0], approx[0][0][1])
        print(approx,approx.shape)
        print(left,right)
        cv2.rectangle(image,left,right,(0,0,255),3)
        cv2.rectangle(image, left2, right2, (0, 255, 255), 3)
        cv2.line(image,left,left2,(255,0,255),3)
        print(len(cnts))
cv2.imwrite("120/ImageAlign/ed2.jpg",warped)
paint(orig,cnts[3])
print(warped.shape)
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)