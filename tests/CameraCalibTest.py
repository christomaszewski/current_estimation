import cv2
import numpy as np

from context import LSPIV_toolkit

import LSPIV_toolkit.vision.calibration as cv_calib

camModel = cv_calib.FisheyeCameraModel()
camModel.loadModel("..\\calib\\GoProHero3Video2.7K.calib")

img = cv2.imread("C:\\Users\\ckt\\Documents\\datasets\\calibration\\GoPro Hero 3 - 2.7K Video\\calib\\frame_567.jpg")

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)

undistorted = camModel.undistortImage(img)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)

cv2.waitKey(0)