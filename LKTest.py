import numpy as np
import matplotlib.pyplot as plt
import vf_utils.vector_field as vf
import vf_utils.approximate as vf_approx
import vf_utils.core as vf_core
import vf_utils.data_viz as vf_viz

import cv_utils.tracker as cv_tracker

import cv2

import glob
import sys

datasetDir = "C:\\Users\\ckt\\Documents\\datasets\\river\\short"

#sys.argv[1]

cv2.namedWindow('img', cv2.WINDOW_NORMAL)

images = glob.glob(datasetDir + "\\*.tiff")
list.sort(images)

vfEstimator = vf_approx.PolynomialLSApproxmiator(2)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


lkTracker = cv_tracker.LKOpticalFlowTracker(lk_params, feature_params)

timestamp = 0.0

for fileName in images:
	img = cv2.imread(fileName)

	print("processing image ", fileName)

	lkTracker.processImage(img, timestamp)

	timestamp += 0.033

	for track in lkTracker.getTracks():
		vfEstimator.addMeasurements(track.getMeasurements())

	field = vfEstimator.approximate()
	viz = img.copy()
	cv2.polylines(viz, [np.int32(tr.getPointSequence()) for tr in lkTracker.getTracks()], False, (255, 0, 0))

	cv2.imshow('img', viz)

	ch = 0xFF & cv2.waitKey(1000)
	if ch == 27:
		break

	#print(lkTracker.getTracks())
