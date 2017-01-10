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

savedDataFilename = '../scenarios/augmented.scenario'
with open(savedDataFilename, mode='rb') as f:
	augmentedVF = dill.load(f)

savedDataFilename2 = '../scenarios/allegheny.scenario'
with open(savedDataFilename2, mode='rb') as f:
	approxVF = dill.load(f)

datasetName = 'boat_left'
datasetDir = '../../../datasets/river/'
dataset = datasetDir + datasetName
images = glob.glob(dataset + '/*.jpg')
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

undistortedImg = None
cropExtents = None
renderTimestep = 1
renderTime = 0
imgCopy = None
for fileName in images:
	print("Processing Image: ", fileName)
	img = cv2.imread(fileName)

	if (not camModel.initialized):
		camModel.initialize(img.shape[:2])
		frameTrans = cv_utils.FrameTransformation(img.shape[:2], camModel)
		cropExtents = frameTrans.getCroppedExtents()


	undistortedImg = frameTrans.transformImg(img)
	if (imgCopy is None):
		imgCopy = undistortedImg.copy()

	result = boatDetector.detect(undistortedImg.copy())

	
	if (result):
		# Flip y coodinates to change from image to field frame
		xPos, yPos = boatDetector.midpoint
		boatTrack.addObservation((xPos, 700-yPos), timeStamp)

	timeStamp += datasetTimestep
	renderTime += datasetTimestep

	ptSeq = boatTrack.getPointSequence()

	interim = np.int32(ptSeq)
	interim[:, 1] = 700 - interim[:, 1]

	cv2.polylines(undistortedImg, [interim], False, (255,0,0))

	cv2.imshow('Input', undistortedImg)

	if (renderTime > renderTimestep):
		renderTime %= 1.0
		cv2.imwrite(subFolder + '/frame_' + str(int(timeStamp)) + '.png', undistortedImg)

	ch = 0xFF & cv2.waitKey(1)
	if ch == 27:
		break


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
augmentedSim = vf_sim.ParticleSimulator(augmentedVF, noise=0)

simulatedTrack = sim.simulate([boatStart], simTime, datasetTimestep)
ptSeq = simulatedTrack[0].getPointSequence()
interim = np.int32(ptSeq)
interim[:, 1] = 700 - interim[:, 1]

cv2.polylines(undistortedImg, [interim], False, (0,0,255))
cv2.polylines(imgCopy, [interim], False, (0,0,255))

simulatedTrack2 = augmentedSim.simulate([boatStart], simTime, datasetTimestep)
ptSeq = simulatedTrack2[0].getPointSequence()
interim = np.int32(ptSeq)
interim[:, 1] = 700 - interim[:, 1]
cv2.polylines(undistortedImg, [interim], False, (0,255,0))


cv2.imshow('Input', undistortedImg)

diff = np.asarray(simulatedTrack2[0] - boatTrack)
diff *= diff

dist = np.sqrt(diff[:, 0] + diff[:, 1])
timeArray = np.arange(len(dist)) + 1

fig = plt.figure(figsize=(14, 10), dpi=100)
ax2 = fig.add_subplot(1,1,1)
ax2.plot(timeArray, dist, color='blue')
ax2.set_title('Distance of Boat to Predicted Track')
ax2.hold(True)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance (px)')
ax2.minorticks_on()
ax2.grid(which='both', alpha=1.0, linewidth=1)
ax2.tick_params(which='both', direction='out')
plt.show()
fig.savefig(subFolder + '/distance_to_predicted.png', bbox_inches='tight', dpi=100)


f = plt.figure(figsize=(14, 10), dpi=100)
ax = f.add_subplot(1,1,1)
ax.set_title('Drift Error')
ax.hold(True)
ax.minorticks_on()
ax.axis([120, 520, 250, 650])
ax.grid(which='both', alpha=1.0, linewidth=1)
ax.tick_params(which='both', direction='out')
aPts = np.asarray(boatTrack.getPointSequence())
sPts = np.asarray(simulatedTrack[0].getPointSequence())


ax.scatter(aPts[:,0], aPts[:,1], c='red', edgecolor='red', marker='o')
ax.scatter(sPts[:,0], sPts[:,1], c='blue', edgecolor='blue', marker='o')

for a,s in zip(boatTrack.getPointSequence()[::2], simulatedTrack[0].getPointSequence()[::2]):
	ax.plot([a[0], s[0]], [a[1], s[1]], c='black')

f.savefig(subFolder + '/drift_error.png', bbox_inches='tight', dpi=100)

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
overlayView.save(subFolder + '/overlay.png')

trackGlob = sorted(glob.glob(subFolder + '/frame_*.png'), key=os.path.getmtime)

images = [imageio.imread(file) for file in trackGlob]
imageio.mimsave(subFolder + '/boatTrack.gif', images, duration=1, loop=2)

cv2.waitKey()
cv2.imwrite(subFolder + '/trackCompare.png', undistortedImg)
cv2.imwrite(subFolder + '/predictedTrack.png', imgCopy)

