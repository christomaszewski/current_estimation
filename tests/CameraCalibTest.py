import cv2
import numpy as np

from context import LSPIV_toolkit

import LSPIV_toolkit.vision.calibration as cv_calib
import LSPIV_toolkit.vision.utils as cv_utils

camModel = cv_calib.FisheyeCameraModel()
camModel.loadModel('../calib/GoProHero3Video2.7K.calib')

img = cv2.imread('../../../datasets/river/boat.tiff')
camModel.initialize(img.shape[:2])
frameTrans = cv_utils.FrameTransformation(img.shape[:2], camModel)


cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)

undistorted = frameTrans.transformImg(img)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)
cv2.imwrite('original.png', img)
cv2.imwrite('undistorted.png', undistorted)

cv2.waitKey(0)