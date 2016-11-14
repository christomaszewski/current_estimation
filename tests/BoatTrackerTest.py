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
import LSPIV_toolkit.sim.simulators as vf_sim

headless = False

savedDataFilename = '../output/lspiv_base/complete/approxVF.scenario'
with open(savedDataFilename, mode='rb') as f:
	approxVF = dill.load(f)

datasetName = 'boat_center_2'
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
undistortedImg = None
cropExtents = None

for fileName in images:
	print("Processing Image: ", fileName)
	img = cv2.imread(fileName)

	if (not camModel.initialized):
		camModel.initialize(img.shape[:2])
		frameTrans = cv_utils.FrameTransformation(img.shape[:2], camModel)
		cropExtents = frameTrans.getCroppedExtents()


	undistortedImg = frameTrans.transformImg(img)
	result = boatDetector.detect(undistortedImg.copy())

	
	if (result):
		# Flip y coodinates to change from image to field frame
		xPos, yPos = boatDetector.midpoint
		boatTrack.addObservation((xPos, 700-yPos), timeStamp)

	timeStamp += datasetTimestep

	ptSeq = boatTrack.getPointSequence()
	"""
	cv2.polylines(undistortedImg, [np.int32(ptSeq)], False, (255,0,0))

	cv2.imshow('Input', undistortedImg)

	ch = 0xFF & cv2.waitKey(1)
	if ch == 27:
		break
	"""

ptSeq = boatTrack.getPointSequence()
interim = np.int32(ptSeq)
interim[:, 1] = 700 - interim[:, 1]

cv2.polylines(undistortedImg, [interim], False, (255,0,0))

# Get Boat start point
boatStartTime, boatStartPoint = boatTrack[0]

# Reset boat start point time
boatStart = (0, (boatStartPoint[0], boatStartPoint[1]))

simTime = boatTrack.age()

sim = vf_sim.ParticleSimulator(approxVF, noise=0)

simulatedTrack = sim.simulate([boatStart], simTime, datasetTimestep)
ptSeq = simulatedTrack[0].getPointSequence()
interim = np.int32(ptSeq)
interim[:, 1] = 700 - interim[:, 1]

cv2.polylines(undistortedImg, [interim], False, (0,0,255))
cv2.imshow('Input', undistortedImg)

xGrid = 25 #cells
yGrid = 15 #cells
xDist = cropExtents[0][1]
yDist = cropExtents[1][1]

grid = vf_core.utils.SampleGrid(xDist, yDist, xGrid, yGrid)

overlayView = vf_plot.OverlayFieldView()
overlayView.setTitle('Current Flow Estimate')
overlayView.changeGrid(grid)
overlayView.updateImage(undistortedImg)
overlayView.changeField(approxVF)
overlayView.quiver()
overlayView.plotTrack(simulatedTrack[0], color='red')

cv2.waitKey()
cv2.imwrite(subFolder + '/trackCompare.png', undistortedImg)
