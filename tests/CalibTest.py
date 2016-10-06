import cv2

import cv_utils.calibration as cv_calib

camModel = cv_calib.FisheyeCameraModel()
camModel.loadModel("calib\\GoProHero3Image.calib")

img = cv2.imread("C:\\Users\\ckt\\Documents\\datasets\\calibration\\GoPro Hero 3 - 12MP Image\\GOPR0189.JPG")

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)

undistorted = camModel.undistortImage(img)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)

cv2.waitKey(0)