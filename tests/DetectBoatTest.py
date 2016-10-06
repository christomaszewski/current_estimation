import cv2
import numpy as np

from context import cv_utils
import cv_utils.detectors as cv_detectors
import cv_utils.calibration as cv_calib

camModel = cv_calib.FisheyeCameraModel()
camModel.loadModel("..\\calib\\GoProHero3Video2.7K.calib")

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
img = cv2.imread("..\\frame_11530.tiff")
#img = cv2.imread("frame_532.jpg")

undistorted = camModel.undistortImage(img, crop=True)

print(undistorted.shape)


detector = cv_detectors.BoatDetector()

#result = detector.detect(undistorted)
#print(result)


cv2.imshow("img", undistorted)
cv2.imwrite("result.tiff", undistorted)
cv2.waitKey(0)