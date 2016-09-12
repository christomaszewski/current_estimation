import cv2
import sys

import cv_utils.calibration as cv_calib

#todo: add help and checking that both parameters have been passed
calibImgDir = sys.argv[1]
calibName = sys.argv[2]

fisheyeCalib = cv_calib.FisheyeCalibration()

fisheyeModel = fisheyeCalib.computeModel(calibImgDir)
# "C:\\Users\\ckt\\Documents\\datasets\\calibration\\GoPro Hero 3 - 12MP Image")

fisheyeModel.saveModel(calibName)