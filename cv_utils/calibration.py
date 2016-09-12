import cv2
import pickle
import glob
import numpy as np

class FisheyeCameraModel(object):
	""" Class for loading calibration settings and rectifying images accordingly

		todo: add warping of individual points
	"""

	def __init__(self, camIntrinsics=None, distortionCoeff=None):
		self.__K = camIntrinsics
		self.__D = distortionCoeff

	def loadModel(self, fileName):
		with open(fileName, 'rb') as f:
			(self.__K, self.__D) = pickle.load(f)

	def saveModel(self, fileName):
		with open(fileName, 'wb') as f:
			pickle.dump((self.__K, self.__D), f)

	def undistortPoints(self, points):
		undistorted = cv2.fisheye.undistortPoints(points, self.__K, self.__D)

	def undistortImage(self, img):
		undistortedSize = img.shape[:2]

		map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.__K, self.__D, np.eye(3),
			self.__K, undistortedSize, cv2.CV_16SC2)

		undistorted = cv2.remap(img, map1, map2,
			interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		#rectified = cv2.remap(img, self.__camIntrinsics[0], self.__camIntrinsics[1],
		#	 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

		return undistorted



class FisheyeCalibration(object):
	""" Class to produce a FisheyeCameraModel from a set of calibration images

		Only standard checkerboard images supported for now
	"""

	def __init__(self, checkerboardDim=(9,6)):
		self.__checkerboardDim = checkerboardDim

		self.__termCrit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.00001)

		# Prepare object points corresponding to calibration checkerboard
		x, y = checkerboardDim
		self.__objPoints = np.zeros((1, x * y, 3), np.float32)
		self.__objPoints[0,:,:2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
		
		# Allocate space for camera intrinsics and distortion coefficients
		self.__K = np.zeros((3,3))
		self.__D = np.zeros((4,1))

	def computeModel(self, imgDir, fileExtension="JPG"):
		# Note Windows path...
		imageFiles = glob.glob(imgDir + "\\*." + fileExtension)

		objPoints = [] # 3d points in real world space
		imgPoints = [] # 2d points in image plane

		for fileName in imageFiles:
			img = cv2.imread(fileName)
			grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			ret, corners = cv2.findChessboardCorners(grayImg, self.__checkerboardDim,
				cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

			if (ret):
				print("Found corners in image", fileName)
				objPoints.append(self.__objPoints)

				# todo: check if window size needs to adjust based on checkerboard
				returnVal = cv2.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), self.__termCrit)
				imgPoints.append(corners.reshape(1, -1, 2))
			else:
				print("Could not find corners in image", fileName)

		numImgPoints = len(imgPoints)
		rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(numImgPoints)]
		tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(numImgPoints)]

		print("Finished processing images, computing model")
		err, _, _, _, _ = cv2.fisheye.calibrate(objPoints, imgPoints, grayImg.shape[::-1],
			self.__K, self.__D, rvecs, tvecs,
			cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW, 
			self.__termCrit)

		fisheyeCam = FisheyeCameraModel(self.__K, self.__D)

		return fisheyeCam