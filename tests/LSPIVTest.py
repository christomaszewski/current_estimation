import cv2
import glob
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

from context import LSPIV_toolkit

import LSPIV_toolkit.vision.calibration as cv_calib
import LSPIV_toolkit.vision.utils as cv_utils
import LSPIV_toolkit.vision.trackers as cv_trackers
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.core.vf.extents as vf_extents
import LSPIV_toolkit.approx as vf_approx
import LSPIV_toolkit.core.plotting as vf_plot
import LSPIV_toolkit.analysis as vf_analysis

datasetName = 'lspiv_base'
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
camModel.loadModel("..\\calib\\GoProHero3Video2.7K.calib")

cv2.namedWindow("Input", cv2.WINDOW_NORMAL)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 600,
                       qualityLevel = 0.5,
                       minDistance = 5,
                       blockSize = 7 )

lkTracker = cv_trackers.LKOpticalFlowTracker(lk_params, feature_params)

timestamp = 0.0

vfEstimator = vf_approx.gp.GPApproximator()
vfExtents = None

xGrid = 25 #cells
yGrid = 15 #cells

displayGrid = None

fieldView = vf_plot.SimpleFieldView()
fieldView.setTitle('Current Flow Estimate')
mFilter = None
frameTrans = None
renderTime = 0.0


for fileName in images:
	print("Processing Image: ", fileName)
	img = cv2.imread(fileName)

	if (frameTrans is None):
		camModel.initialize(img.shape[:2])
		frameTrans = cv_utils.FrameTransformation(img.shape[:2], camModel)
	
	if (vfExtents is None):
		cropExtents = frameTrans.getCroppedExtents()
		vfExtents = vf_extents.FieldExtents(cropExtents[0], cropExtents[1])
		xDist = cropExtents[0][1]
		yDist = cropExtents[1][1]
		print("cropped img size: ", xDist, yDist)
		displayGrid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)
		fieldView.changeGrid(displayGrid)
		mFilter = vf_analysis.MeasurementProcessor(xDist, yDist, xGrid, yGrid)

	undistortedImg = camModel.undistortImage(img)

	lkTracker.processImage(img, timestamp)
	timestamp += datasetTimestep
	renderTime += datasetTimestep

	if (renderTime < renderTimestep):
		continue

	renderTime %= renderTimestep

	tracks = lkTracker.getTracks()

	undistortedImg = frameTrans.transformImg(img)
	b,g,r = cv2.split(undistortedImg)
	undistortedImgColor = cv2.merge([r,g,b])

	for tr in tracks:
		if (tr.age() < minTrackAge):
			continue
		# For Plotting only
		track = frameTrans.transformTrackForPlotting(tr)

		newTrack = frameTrans.transformTrack(tr)

		ptSeq = track.getPointSequence()
		cv2.polylines(undistortedImg, [np.int32(ptSeq)], False, (255,0,0))
		
		mFilter.addMeasurements(newTrack.getMeasurements())
		#print(len(mFilter.getMeasurements()))
		#vfEstimator.addMeasurements(mFilter.getMeasurements())

	cv2.imshow("Input", undistortedImg)
	outputFile = subFolder + '/img_' + str(int(timestamp)) + '.png'
	cv2.imwrite(outputFile, undistortedImg)

	approxFileName = subFolder + '/approx_' + str(int(timestamp)) + '.png'




	vfEstimator.clearMeasurements()
	vfEstimator.addMeasurements(mFilter.getMeasurements())
	mFilter.drawMeasurementGrid()
	vfApprox = vfEstimator.approximate(vfExtents)

	xgrid, ygrid = displayGrid.mgrid
	xSamples, ySamples = vfApprox.sampleAtGrid(xgrid, ygrid)
	magnitudes = np.sqrt(xSamples**2 + ySamples**2)

	clim = [magnitudes.min(), magnitudes.max()]

	f = plt.figure()
	ax = f.add_subplot(1,1,1)
	
	q = plt.quiver(xgrid, ygrid, xSamples, ySamples, magnitudes,
				clim=clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)
	plt.hold(True)
	
	undistortedImgColor = np.flipud(undistortedImgColor)
	plt.imshow(undistortedImgColor, origin='lower', aspect='auto')
	f.colorbar(q, ax=ax)
	plt.show()
	overlayFile = subFolder + '/overlay_' + str(int(timestamp)) + '.png'
	plt.savefig()
	plt.close()


	fieldView.changeField(vfApprox)
	fieldView.save(approxFileName)


	ch = 0xFF & cv2.waitKey(10)
	if ch == 27:
		break