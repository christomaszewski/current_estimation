import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from context import LSPIV_toolkit

import LSPIV_toolkit.vision.trackers as cv_trackers


datasetDir = "C:\\Users\\ckt\\Documents\\datasets\\river\\short"

cv2.namedWindow('img', cv2.WINDOW_NORMAL)

images = glob.glob(datasetDir + "\\*.tiff")
list.sort(images)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 7 )


lkTracker = cv_trackers.LKOpticalFlowTracker(lk_params, feature_params)

timestamp = 0.0

for fileName in images:
	img = cv2.imread(fileName)
	print(img.shape)

	print("processing image ", fileName)

	lkTracker.processImage(img, timestamp)

	timestamp += 0.033

	viz = img.copy()
	cv2.polylines(viz, [np.int32(tr.getPointSequence()) for tr in lkTracker.getTracks()], False, (255, 0, 0))

	cv2.imshow('img', viz)

	ch = 0xFF & cv2.waitKey(1)
	if ch == 27:
		break
