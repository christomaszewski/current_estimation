import cv2
import numpy as np

class GridFeatureDetector(object):
	""" Class to run feature detector on an image in order to generate uniform
		number of measurements in each grid cell across image. Used to deal with
		feature rich regions hoarding entire feature quota

		Currently runs very slow - needs better structure/optimization

	"""

	def __init__(self, detector, gridDim=(2,2), partitionMethod='subimage'):
		# only supports cv2.goodFeaturesToTrack right now
		# partitionMethod subimage is fastest method
		# todo: add support for ORB
		self.__featureDetector = detector

		partitionFuncName = '_' + partitionMethod
		self.__partitionFunc = getattr(self, partitionFuncName, self._nopartition)

		self.setGrid(gridDim)

	def setGrid(self, gridDim):
		self.__gridDimensions = gridDim
		self.__numCells = gridDim[0] * gridDim[1]

	def detect(self, img, mask, params, numFeatures=None):
		if ('maxCorners' in params):
			# goodFeaturesToTrack parameter
			self.__numTotalFeatures = params['maxCorners']
		elif (numFeatures is not None):
			self.__numTotalFeatures = numFeatures
		else:
			# Default number of features
			self.__numTotalFeatures = 1000

		self.__numFeaturesPerCell = int(self.__numTotalFeatures / self.__numCells)

		heightStep = int(img.shape[0] / self.__gridDimensions[0])
		widthStep = int(img.shape[1] / self.__gridDimensions[1])

		paramCopy = params.copy()

		features = self.__partitionFunc(img, mask, paramCopy, heightStep, widthStep)

		return features

	# Partition Functions
	def _mask(self, img, mask, params, heightStep, widthStep):
		# Set desired number of features to features per cell
		params['maxCorners'] = self.__numFeaturesPerCell

		# Declare features array
		features = None

		# Define full mask
		searchMask = np.zeros_like(mask)

		for i in np.arange(0, img.shape[0], heightStep):
			for j in np.arange(0, img.shape[1], widthStep):
				searchMask[i:(i+heightStep), j:(j+widthStep)] = mask[i:(i+heightStep), j:(j+widthStep)]

				newFeatures = self.__featureDetector(img, mask=searchMask, **params)

				# Reset search mask 
				searchMask[i:(i+heightStep), j:(j+widthStep)] = 0

				if (features is None):
					features = newFeatures
				elif (newFeatures is not None):
					features = np.concatenate((features, newFeatures))

		return features

	
	def _subimage(self, img, mask, params, heightStep, widthStep):
		# Set desired number of features to features per cell
		params['maxCorners'] = self.__numFeaturesPerCell

		# Declare features array
		features = None

		for i in np.arange(0, img.shape[0], heightStep):
			for j in np.arange(0, img.shape[1], widthStep):
				searchMask = mask[i:(i+heightStep), j:(j+widthStep)]
				subImg = img[i:(i+heightStep), j:(j+widthStep)]

				newFeatures = self.__featureDetector(subImg, mask=searchMask, **params)

				if (newFeatures is not None):
					newFeatures += np.array([j, i])

					if (features is None):
						features = newFeatures
					else:
						features = np.concatenate((features, newFeatures))

		return features

	def _nopartition(self, img, mask, params, heightStep, widthStep):
		params['maxCorners'] = self.__numTotalFeatures

		features = self.__featureDetector(img, mask=mask, **params)

		return features


class BoatDetector(object):
	""" Detector for boat with Blue painted front plate and Red painted back plate

		todo: produce track object 
	"""

	def __init__(self):
		self.lowerBlue = np.array([100,50,50], np.uint8)
		self.upperBlue = np.array([110,255,255], np.uint8)
		self.lowerRed = np.array([0,50,50], np.uint8)
		self.upperRed = np.array([10,255,255], np.uint8)
		self.midpoint  = (0,0)

	def detect(self, img):
		# Convert image to HSV space
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Find image regions which fall into blue and red ranges
		blueMask = cv2.inRange(hsv, self.lowerBlue, self.upperBlue)
		redMask = cv2.inRange(hsv, self.lowerRed, self.upperRed)

		# Erode and dilate mask to remove spurious detections
		blueMask = cv2.erode(blueMask, None, iterations=2)
		blueMask = cv2.dilate(blueMask, None, iterations=2)

		# Find blue contours
		blueContours = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		
		# If no blue contours find, terminate (Boat not in frame)
		if (len(blueContours) < 1):
			return False

		# Choose largest blue contour
		maxBlue = max(blueContours, key=cv2.contourArea)

		# Find centroid of largest blue contour
		blueMoments = cv2.moments(maxBlue)
		blueCentroid = (int(blueMoments["m10"] / blueMoments["m00"]), int(blueMoments["m01"] / blueMoments["m00"]))
		
		# Repeat above process for red contours
		redMask = cv2.erode(redMask, None, iterations=2)
		redMask = cv2.dilate(redMask, None, iterations=2)

		# Find red contours and choose largest
		redContours = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

		# If no red contours find, terminate (Boat not in frame)
		if (len(redContours) < 1):
			return False

		# Choose largest red contour
		maxRed = max(redContours, key=cv2.contourArea)

		# Find centroid of largest red contour
		redMoments = cv2.moments(maxRed)
		redCentroid = (int(redMoments["m10"] / redMoments["m00"]), int(redMoments["m01"] / redMoments["m00"]))
		
		dist = np.sqrt((redCentroid[0]-blueCentroid[0])**2+(redCentroid[1]-blueCentroid[1])**2)
		print(dist)


		# If distance between centroids of detected panels matches boat report detection
		if (dist >= 40 and dist <= 60):
			cv2.drawContours(img,maxBlue,-1,(0,0,255),3)
			cv2.drawContours(img,maxRed,-1,(255,0,0),3)

			cv2.circle(img, redCentroid, 5, (0,0,0), thickness=-1, lineType=8, shift=0)
			cv2.circle(img, blueCentroid, 5, (0,0,0), thickness=-1, lineType=8, shift=0)
			cv2.line(img, redCentroid, blueCentroid, (0,0,0), thickness=2)

			rise = redCentroid[0]-blueCentroid[0]
			run = redCentroid[1]-blueCentroid[1]

			self.midpoint = (int(blueCentroid[0]+rise*0.5), int(blueCentroid[1]+run*0.5))
			cv2.circle(img, self.midpoint, 5, (0,0,0), thickness=-1)
			print(self.midpoint)
			return True

		return False