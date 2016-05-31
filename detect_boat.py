import cv2
import numpy as np
import pickle
import glob
from detectors import BoatDetector
import sys
from collections import deque

boatTrack = deque(maxlen=200)

with open('cameracalib', 'rb') as f:
	mappingTuple = pickle.load(f)

datasetDir = sys.argv[1]
cv2.namedWindow('img', cv2.WINDOW_NORMAL)


images = glob.glob(datasetDir + "/*.JPG")
list.sort(images)

for fileName in images:

	img = cv2.imread(fileName)

	#img = cv2.imread('boat.JPG')
	undistorted = cv2.remap(img, mappingTuple[0], mappingTuple[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	detector = BoatDetector()

	result = detector.detect(undistorted)
	print(result)

	if (result):
		boatTrack.appendleft(detector.midpoint)

	cv2.imshow('img', undistorted)
	cv2.waitKey(1)

finalImage = cv2.imread(images[50])
undistorted = cv2.remap(finalImage, mappingTuple[0], mappingTuple[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

maxVelocity = 260

# Draw first boat point
prevPoint = boatTrack.popleft()
cv2.circle(undistorted, prevPoint, 5, (255,255,255), thickness=-1)
print(prevPoint)

for point in boatTrack:
	print(point)
	# Compute velocity from previous point assuming 2Hz framerate
	distance = np.sqrt((prevPoint[0]-point[0])**2+(prevPoint[1]-point[1])**2)
	velocity = distance/0.5
	print(velocity)
	mappedVelocity = int(np.interp(velocity, [40, maxVelocity], [0, 80]))
	hue = mappedVelocity
	hsvColor = np.uint8([[[hue,255,255]]])
	clr = cv2.cvtColor(hsvColor, cv2.COLOR_HSV2BGR)
	print(tuple(clr[0][0]))
	b = int(clr[0][0][0])
	g = int(clr[0][0][1])
	r = int(clr[0][0][2])
	cv2.line(undistorted, prevPoint, point, (b,g,r), thickness=10)
	cv2.circle(undistorted, point, 5, (0,0,0), thickness=-1)
	prevPoint = tuple(point)

cv2.imshow('img', undistorted)
cv2.waitKey()
cv2.imwrite('result.png', undistorted)