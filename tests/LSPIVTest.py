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
import LSPIV_toolkit.vision.trackers as cv_trackers
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.core.vf.extents as vf_extents
import LSPIV_toolkit.approx as vf_approx
import LSPIV_toolkit.core.plotting as vf_plot
import LSPIV_toolkit.analysis as vf_analysis

headless = False

datasetName = 'lspiv_test'
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

overlayView = vf_plot.OverlayFieldView()
overlayView.setTitle('Current Flow Estimate')

mFilter = None
frameTrans = None
renderTime = 0.0

font = cv2.FONT_HERSHEY_SIMPLEX

for fileName in images:
	print("Processing Image: ", fileName)
	img = cv2.imread(fileName)

	# Need to pull size of image from first image. Todo: save this to the calib file
	if (frameTrans is None):
		camModel.initialize(img.shape[:2])
		frameTrans = cv_utils.FrameTransformation(img.shape[:2], camModel)
	
	if (vfExtents is None):
		cropExtents = frameTrans.getCroppedExtents()
		vfExtents = vf_extents.FieldExtents(cropExtents[0], cropExtents[1])
		
		xDist = cropExtents[0][1]
		yDist = cropExtents[1][1]
		
		displayGrid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)
		fieldView.changeGrid(displayGrid)
		overlayView.changeGrid(displayGrid)

		mFilter = vf_analysis.MeasurementProcessor(xDist, yDist, xGrid, yGrid)

	lkTracker.processImage(img, timestamp)
	timestamp += datasetTimestep
	renderTime += datasetTimestep

	# Skip rendering/writing output files if it is not time
	if (renderTime < renderTimestep):
		continue

	renderTime %= renderTimestep

	# Unwarp image and set as background of overlayView
	undistortedImg = frameTrans.transformImg(img)
	overlayView.updateImage(undistortedImg)

	# Get latest tracks from Optical flow
	tracks = lkTracker.getTracks()

	for tr in tracks:
		# Skip tracks that are too young
		if (tr.age() < minTrackAge):
			continue

		# For Plotting only
		track = frameTrans.transformTrackForPlotting(tr)

		newTrack = frameTrans.transformTrack(tr)

		ptSeq = track.getPointSequence()
		cv2.polylines(undistortedImg, [np.int32(ptSeq)], False, (255,0,0))
		
		mFilter.addMeasurements(newTrack.getMeasurements())

	# Compute Approximation
	vfEstimator.clearMeasurements()
	vfEstimator.addMeasurements(mFilter.getMeasurements())
	approxVF = vfEstimator.approximate(vfExtents)

	# Define output file names
	timeString = str(int(timestamp))
	outputFile = subFolder + '/img_' + timeString + '.png'
	approxFile = subFolder + '/approx_' + timeString + '.png'
	overlayFile = subFolder + '/overlay_' + timeString + '.png'
	measurementFile = subFolder + '/measurements_' + timeString + '.png'

	# Update Annotations
	annoation = 'Time: ' + timeString
	overlayView.setAnnotation(' (' + annoation + ')')
	fieldView.setAnnotation(' (' + annoation + ')')
	mFilter.setAnnotation(' (' + annoation + ')')
	cv2.putText(undistortedImg, annoation, (500,100), font, 1, 
				(255,255,255), 2, cv2.LINE_AA)

	# Update visuals
	if (not headless):
		cv2.imshow("Input", undistortedImg)
	overlayView.changeField(approxVF)
	fieldView.changeField(approxVF)
	mFilter.drawMeasurementGrid()

	# Save output files
	overlayView.save(overlayFile)
	fieldView.save(approxFile)
	mFilter.saveFig(measurementFile)
	cv2.imwrite(outputFile, undistortedImg)


# Process output images and generate gifs
tracksGlob = sorted(glob.glob(subFolder + '/img_*.png'), key=os.path.getmtime)
measurementGlob = sorted(glob.glob(subFolder + '/measurements_*.png'), key=os.path.getmtime)
approxGlob = sorted(glob.glob(subFolder + '/approx_*.png'), key=os.path.getmtime)
overlayGlob = sorted(glob.glob(subFolder + '/overlay_*.png'), key=os.path.getmtime)

images = [imageio.imread(file) for file in measurementGlob]
imageio.mimsave(subFolder + '/measurements.gif', images, duration=1, loop=2)

images = [imageio.imread(file) for file in tracksGlob]
imageio.mimsave(subFolder + '/tracks.gif', images, duration=1, loop=2)

images = [imageio.imread(file) for file in approxGlob]
imageio.mimsave(subFolder + '/approx.gif', images, duration=1, loop=2)

images = [imageio.imread(file) for file in overlayGlob]
imageio.mimsave(subFolder + '/overlay.gif', images, duration=1, loop=2)

# Save final approximation to disk
with open(subFolder + '/approxVf.scenario', mode='wb') as f:
	dill.dump(approxVF, f)