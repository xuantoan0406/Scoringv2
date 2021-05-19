import cv2
import numpy as np
def canny_edge_detection(image_path, blur_ksize=3, threshold1=25, threshold2=200):
	"""
	image_path: link to image
	blur_ksize: Gaussian kernel size
	threshold1: min threshold
	threshold2: max threshold
	"""
	img = cv2.imread(image_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
	img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)
	
	return img_canny


# read image
# img = cv2.imread('sugoku.jpg')
# # convert to gray scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # color -> gray
# edges = cv2.Canny(gray, 80, 150, apertureSize=3)
# cv2.imshow("a",edges)
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 300*(-b))
#     y1 = int(y0 + 300*(a))
#     x2 = int(x0 - 300*(-b))
#     y2 = int(y0 - 300*(a))
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# cv2.imshow('geo_hough.jpg',img)
# cv2.waitKey()

# img = cv2.imread("board.png")
#
# cv2.rectangle(img,(0,0),(50,50),(0,0,255),3)
# cv2.imshow("a",img)
# cv2.waitKey()
# img = cv2.imread("120/l1.jpg")
# thresh=canny_edge_detection("120/l1.jpg")
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
#
# contours_sizes= [(cv2.contourArea(cnt), cnt) for cnt in contours]
# biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]
#
# countours = biggest_contour
# cv2.drawContours(img, countours, -1, (0, 255, 255), 3)
#
# print(countours)
# cv2.imshow("a",img)
# cv2.waitKey()
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped
