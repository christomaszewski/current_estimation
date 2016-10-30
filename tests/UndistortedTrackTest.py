import cv2
import glob
import numpy as np

from context import LSPIV_toolkit

import LSPIV_toolkit.vision.calibration as cv_calib
import LSPIV_toolkit.vision.trackers as cv_trackers

datasetDir = "C:\\Users\\ckt\\Documents\\datasets\\river\\short"
images = glob.glob(datasetDir + "\\*.tiff")
list.sort(images)
datasetTimestep = 0.033 # 29.9 FPS

camModel = cv_calib.FisheyeCameraModel()
camModel.loadModel("..\\calib\\GoProHero3Video2.7K.calib")

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)

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
	undistortedImg = camModel.undistortImage(img, 'extents')

	lkTracker.processImage(img, timestamp)
	timestamp += datasetTimestep

	tracks = lkTracker.getTracks()

	for tr in tracks:
		ptSeq = tr.getPointSequence()
		cv2.polylines(img, [np.int32(ptSeq)], False, (255,0,0))
		undistortedPtSeq = camModel.undistortPoints(np.asarray(ptSeq))

		undistortedPtSeq = np.reshape(undistortedPtSeq, (-1,2))

		# Hack to make tracks display properly in cropped image - fix this!
		xOffset, yOffset, _, _ = camModel._cropBounds
		undistortedPtSeq[:, 0] -= xOffset
		undistortedPtSeq[:, 1] -= yOffset
		
		cv2.polylines(undistortedImg, [np.int32(undistortedPtSeq)], False, (0,0,255))

	
	cv2.imshow("Original", img)
	cv2.imshow("Undistorted", undistortedImg)

	ch = 0xFF & cv2.waitKey(10)
	if ch == 27:
		cv2.imwrite("original.png", img)
		cv2.imwrite("undistorted.png", undistortedImg)
		break


