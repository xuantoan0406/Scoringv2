import cv2
from matplotlib import pyplot as plt
import numpy as np
import edgeDetect as edge
import imutils
import matplotlib.image as mpimg
def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        print(top_r, top_l, bottom_l, bottom_r)
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype = "float32")
    


    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))
#a='t2.jpg'
a='board.png'
image = cv2.imread(a)
original = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 3)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)
# cv2.imshow("a",thresh)
# cv2.imwrite("2bit.jpg",thresh)
thresh=edge.canny_edge_detection(a)
cv2.imwrite("2bit.jpg",thresh)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
j=0
for contour in cnts:
    if cv2.contourArea(contour) > 1000:
        cv2.drawContours(image, contour, -1, (0, 255, 255), 3)
        j+=1
        print("....j= {}..............{}".format(j,contour.shape))

#cv2.drawContours(image,cnts,-1,(255,25,25),5)



def sortCoordinates(approx):
    if len(approx)==4:
        return approx
    else:
        return None
i=0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    
    i+=1
    print(approx.shape)
    if len(approx) == 4:
        
        screenCnt = approx
        print("......{}....".format(i))
        break
    #approx=sortCoordinates(approx)
    
transformed = perspective_transform(original, screenCnt)



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
paint(original,cnts[6])
cv2.imwrite('board1.png', transformed)

cv2.imshow('transformed', imutils.resize(image, height = 1000))
cv2.waitKey()

