import cv2
import glob
import numpy as np

from context import LSPIV_toolkit
import LSPIV_toolkit.vision.detectors as cv_detectors


datasetDir = "C:\\Users\\ckt\\Documents\\datasets\\river\\test"

cv2.namedWindow('imgPartitioned', cv2.WINDOW_NORMAL)
cv2.namedWindow('imgFull', cv2.WINDOW_NORMAL)

images = glob.glob(datasetDir + "\\*.tiff")
list.sort(images)

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 7)

for image in images:

	img = cv2.imread(image)
	img1 = img.copy()
	img2 = img.copy()
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	searchMask = np.zeros_like(grayImg)
	searchMask[:] = 255
	gridDetector = cv_detectors.GridFeatureDetector(cv2.goodFeaturesToTrack, (10,10))

	p = cv2.goodFeaturesToTrack(grayImg, mask=searchMask, **feature_params)

	p1 = gridDetector.detect(grayImg, searchMask, feature_params)


	for point in enumerate(p):
		a, b  = point[1].ravel()
		cv2.circle(img, (a,b), 5, (0, 0, 255), -1)

	for point in enumerate(p1):
		#print(p1, point)
		a, b  = point[1].ravel()
		cv2.circle(img1, (a,b), 5, (255, 0, 0), -1)


	cv2.imshow('imgFull', img)
	cv2.imshow('imgPartitioned', img1)
	cv2.waitKey(0)