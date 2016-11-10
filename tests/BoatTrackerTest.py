import cv2
import glob
import numpy as np
import os
import time
import matplotlib as mpl
# Uncomment this to run headless/remove tkinter dependency
#mpl.use('agg')
import matplotlib.pyplot as plt
import imageio
import dill

from context import LSPIV_toolkit

import LSPIV_toolkit.vision.calibration as cv_calib
import LSPIV_toolkit.vision.utils as cv_utils
import LSPIV_toolkit.vision.detectors as cv_detectors
import LSPIV_toolkit.core as vf_core
import LSPIV_toolkit.core.vf.extents as vf_extents
import LSPIV_toolkit.approx as vf_approx
import LSPIV_toolkit.core.plotting as vf_plot
import LSPIV_toolkit.analysis as vf_analysis

headless = False

datasetName = 'boat_right'
datasetDir = '../../../datasets/river/'
dataset = datasetDir + datasetName
images = glob.glob(dataset + '/*.tiff')
list.sort(images)

outputDir = '../output/' + datasetName

if not os.path.exists(outputDir):
	os.makedirs(outputDir)

# Generate unique run folder
subFolder =  outputDir + '/' + time.strftime('%d_%m_%Y_%H_%M_%S')

if not os.path.exists(subFolder):
	os.makedirs(subFolder)

datasetTimestep = 0.033 # 29.9 FPS
renderTimestep = 1
minTrackAge = 0.5

camModel = cv_calib.FisheyeCameraModel()
camModel.loadModel('../calib/GoProHero3Video2.7K.calib')

if (not headless):
	cv2.namedWindow("Input", cv2.WINDOW_NORMAL)

boatDetector = cv_detectors.BoatDetector()

boatTrack = vf_core.tracking.Track()

timeStamp = 0.0

frameTrans = None
roi = None
hsv_roi = None
roi_hist = None
mask = None
trackWindow = None
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

for fileName in images:
	print("Processing Image: ", fileName)
	img = cv2.imread(fileName)

	if (not camModel.initialized):
		camModel.initialize(img.shape[:2])
		frameTrans = cv_utils.FrameTransformation(img.shape[:2], camModel)


	undistortedImg = frameTrans.transformImg(img)
	result = boatDetector.detect(undistortedImg.copy())

	
	if (result):
		boatTrack.addObservation(boatDetector.midpoint, timeStamp)

	"""
	if (result and roi is None):
		roi = boatDetector.getROI(undistortedImg)
		x, y, w, h = cv2.boundingRect(roi)
		trackWindow = (x, y, w, h)
		roi_img = undistortedImg[y:y+h, x:x+w]
		hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
		mask_roi = mask[y:y+h, x:x+w]
		roi_hist = cv2.calcHist([hsv_roi],[0, 1],mask_roi,[180, 256],[0,180, 0, 256])
		roi_hist.reshape(-1)
		cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

	(x, y, w, h) = trackWindow
	#rect = cv2.minAreaRect(roi)
	#box = cv2.boxPoints(rect)
	
	hsv = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,180, 0, 256],1)

	# apply meanshift to get the new location
	ret, trackWindow = cv2.CamShift(dst, trackWindow, term_crit)

	# Draw it on image
	pts = cv2.boxPoints(ret)
	pts = np.int0(pts)
	x, y, w, h = trackWindow
	#cv2.polylines(undistortedImg,[pts],True, 255,2)
	cv2.rectangle(undistortedImg, (x,y), (x+w,y+h), 255,2)
	"""
	timeStamp += datasetTimestep

	ptSeq = boatTrack.getPointSequence()
	cv2.polylines(undistortedImg, [np.int32(ptSeq)], False, (255,0,0))

	cv2.imshow('Input', undistortedImg)

	ch = 0xFF & cv2.waitKey(1)
	if ch == 27:
		break
