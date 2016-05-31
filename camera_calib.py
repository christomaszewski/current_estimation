import numpy
import cv2
import glob
import pickle


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = numpy.zeros((1,6*9,3), numpy.float32)
objp[0,:,:2] = numpy.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('~/Documents/datasets/calib/*.JPG')
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

for fname in images:
	print("loading image " + fname)
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow('img', gray)
	cv2.waitKey(250)

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
	# If found, add object points, image points (after refining them)
	if ret == True:
		print("Found corners")
		objpoints.append(objp)

		cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners.reshape(1,-1,2))

		# Draw and display the corners
		cv2.drawChessboardCorners(img, (9,6), corners,ret)
		cv2.imshow('img',img)
		cv2.waitKey(250)


N_OK = len(imgpoints)
rvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float64) for i in range(N_OK)]
tvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float64) for i in range(N_OK)]
K = numpy.zeros((3,3))
D = numpy.zeros((4,1))

err, _,_, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs, cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW, criteria)

img = cv2.imread('calib/GOPR0189.JPG')

print(img.shape[:2])
#undistorted = cv2.fisheye.undistortImage(img, K, D)

parallelProjectionMatrix = numpy.zeros((4,3))
parallelProjectionMatrix[0,0] = 1
parallelProjectionMatrix[1,1] = 1
parallelProjectionMatrix[3,2] = 1

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, numpy.eye(3), K, (4000,3000), cv2.CV_16SC2)

mappingTuple = (map1, map2)

with open('cameracalib', 'wb') as f:
	pickle.dump(mappingTuple, f)

undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow('img',undistorted)
cv2.waitKey()
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#img = cv2.imread('test/GOPR0207.JPG')
#h,  w = img.shape[:2]
#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#cv2.imshow('img', dst)
#cv2.waitKey()

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult2.png',dst)

cv2.destroyAllWindows()