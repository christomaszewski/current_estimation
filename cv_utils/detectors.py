import cv2
import numpy as np

class UniformGridFeatureDetector(object):
	""" Class to run feature detector on an image in order to generate uniform
		number of measurements in each grid cell across image. Used to deal with
		feature rich regions hoarding entire feature quota

	"""

	def __init__(self, detector, mask, params, gridDim, numFeatures=None):
		# only supports cv2.goodFeaturesToTrack right now
		# numFeatures overrides equivalent parameter in params if set 
		# todo: add support for ORB
		self.__featureDetector = detector
		self.__detectorParams = params
		self.__searchMask = mask
		self.__numFeatures = numFeatures

		self.__gridDimensions = gridDim

		self.__numCells = gridDim[0] * gridDim[1]

		self.__featuresPerCell = int(numFeatures / self.__numCells)


	def detect(self, img):
		features = np.array([])

		width = img.shape[1]
		height = img.shape[0]
		heightStep = int(height / self.__gridDimensions[0])
		widthStep = int(width / self.__gridDimensions[1])

		self.__detectorParams['maxCorners'] = self.__featuresPerCell
		#print(width, height, widthStep, heightStep)
		for i in np.arange(0, height, heightStep):
			for j in np.arange(0, width, widthStep):
				#print(i, j)
				subMask = np.zeros_like(self.__searchMask)
				subMask[i:(i+heightStep), j:(j+widthStep)] = self.__searchMask[i:(i+heightStep), j:(j+widthStep)]
				subImg = img
				#subMask = self.__searchMask[i:(i+heightStep), j:(j+widthStep)]
				#subImg = img[i:(i+heightStep), j:(j+widthStep)]
				newFeatures = self.__featureDetector(subImg, mask=subMask, **self.__detectorParams)

				if (features is not None and features.size < 1):
					# No previously found features
					features = newFeatures
				elif (newFeatures is not None):
					# New features found
					#print(newFeatures)

					#newFeatures += np.array([i, j])
					#print(newFeatures)
					features = np.concatenate((features, newFeatures))

		return features

class BoatDetector(object):

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