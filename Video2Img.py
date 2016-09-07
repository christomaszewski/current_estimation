import sys
import cv2

videoFile = sys.argv[1]
outputDir = sys.argv[2]


cap = cv2.VideoCapture(videoFile)

frameIndex = 0

while (cap.isOpened()):
	ret, frame = cap.read()

	fileName = outputDir + "\\frame_" + str(frameIndex) + ".jpg"

	cv2.imwrite(fileName, frame)

	frameIndex += 1

cap.release()