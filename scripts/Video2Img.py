import sys
import cv2
import time
from imutils.video import FileVideoStream

videoFile = sys.argv[1]
outputDir = sys.argv[2]

fvs = FileVideoStream(videoFile).start()
time.sleep(1.0)

frameIndex = 0

while (fvs.more()):
	frame = fvs.read()

	cv2.imshow("Frame", frame)
	fileName = outputDir + "\\frame_" + str(frameIndex) + ".tiff"
	cv2.imwrite(fileName, frame)

	frameIndex += 1

	cv2.waitKey(1)

cv2.destroyAllWindows()
fvs.stop()