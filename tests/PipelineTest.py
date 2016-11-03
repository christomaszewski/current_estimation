import cv2
import glob
import numpy as np

from context import LSPIV_toolkit

import LSPIV_toolkit.vision.calibration as cv_calib
import LSPIV_toolkit.vision.utils as cv_utils
import LSPIV_toolkit.vision.trackers as cv_trackers
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.core.vf.extents as vf_extents
import LSPIV_toolkit.approx as vf_approx
import LSPIV_toolkit.core.plotting as vf_plot
import LSPIV_toolkit.analysis as vf_analysis

datasetDir = "C:\\Users\\ckt\\Documents\\datasets\\river\\short"
images = glob.glob(datasetDir + "\\*.tiff")
list.sort(images)
datasetTimestep = 0.033 # 29.9 FPS

camModel = cv_calib.FisheyeCameraModel()
camModel.loadModel("..\\calib\\GoProHero3Video2.7K.calib")

cv2.namedWindow("Input", cv2.WINDOW_NORMAL)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 7 )

lkTracker = cv_trackers.LKOpticalFlowTracker(lk_params, feature_params)

timestamp = 0.0

vfEstimator = vf_approx.gp.GPApproximator()
vfExtents = None

xGrid = 25 #cells
yGrid = 10 #cells
xDist = 100 #meters
yDist = 50 #meters

displayGrid = None

fieldView = vf_plot.SimpleFieldView()
mFilter = None
frameTrans = None
for fileName in images:
	img = cv2.imread(fileName)

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
		mFilter = vf_analysis.MeasurementProcessor(xDist, yDist, xGrid, yGrid)

	undistortedImg = camModel.undistortImage(img)

	lkTracker.processImage(img, timestamp)
	timestamp += datasetTimestep

	tracks = lkTracker.getTracks()

	undistortedImg = frameTrans.transformImg(img)

	for tr in tracks:
		# For Plotting only
		track = frameTrans.transformTrackForPlotting(tr)

		newTrack = frameTrans.transformTrack(tr)

		ptSeq = track.getPointSequence()
		cv2.polylines(undistortedImg, [np.int32(ptSeq)], False, (255,0,0))
		mFilter.addMeasurements(newTrack.getMeasurements())
		#print(len(mFilter.getMeasurements()))
		#vfEstimator.addMeasurements(mFilter.getMeasurements())

	vfEstimator.clearMeasurements()
	vfEstimator.addMeasurements(mFilter.getMeasurements())
	mFilter.drawMeasurementGrid()
	vfApprox = vfEstimator.approximate(vfExtents)

	fieldView.changeField(vfApprox)

	cv2.imshow("Input", undistortedImg)

	ch = 0xFF & cv2.waitKey(10)
	if ch == 27:
		break