import cv2
import time

from context import LSPIV_toolkit

import LSPIV_toolkit.core.dataset as data_utils


cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

d = data_utils.Dataset.from_file('data.data')
d.load()

time.sleep(1.0)

while(d.more()):
	cv2.imshow("Image", d.read())
	cv2.waitKey(1)
