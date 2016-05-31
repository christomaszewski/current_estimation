import cv2
import numpy as np

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