import sys
import cv2
#from data_viz import FieldView
import numpy as np
from numpy import linalg as LA
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from functools import reduce
from utils import VectorField,SampleGrid,VectorFieldApproximator


def generateMonomialVector(degree):
	""" Returns a lambda function that can generate the monomial vector for any inputs x and y

	"""

	vectorFunc = lambda point: np.asarray([[point[0]**xExponent * point[1]**yExponent for yExponent in range(0,degree+1) for xExponent in range(0,degree+1-yExponent)]]).transpose()

	return vectorFunc

	#vectorFunc = lambda x,y: np.asarray([x for i in range()])


#img = cv2.imread(sys.argv[1])

#viz = FieldView(img)

#viz.show()

vMax = 3 #m/s
xGrid = 10 #cells
yGrid = 10 #cells
xDist = 10 #meters
yDist = 10 #meters

yCell = yDist/yGrid #meters
yCellHalf = yCell/2.0 #meters
xCell = xDist/xGrid #meters
xCellHalf = xCell/2.0 #meters

riverWidth = 10 #meters

vfFunc = lambda x,y: (0, (4*x/riverWidth - 4*x**2/riverWidth**2)*vMax)
vfBounds = [0, riverWidth, 0, riverWidth]
sourceVF = VectorField(vfFunc, vfBounds)

grid = SampleGrid(xDist, yDist, xGrid, yGrid)

#field = np.asarray(F(x,y) for x in range ())
plt.close('all')
fig = plt.figure()

field = sourceVF.sampleGrid(grid)
gridX, gridY = grid.mgrid

ax1 = fig.add_subplot(311)
sourceVF.quiver(ax1, grid)
ax1.axis([-2 * xCell, xDist + 2 * xCell, -2 * yCell, yDist + 2 * yCell])
#plt.show()


polyDegree = 2
monomialLength = int((polyDegree+1)*(polyDegree+2)/2)

#print([(i,k) for k in range(0,n+1) for i in range(0,n+1-k)])

w = generateMonomialVector(polyDegree)
#points = [(3,3), (5,8), (9,1)]
#points = [(1,1), (3,1), (1,5), (3,5)]
points = [(1,1), (3,1), (5,1), (7,1), (9,1)]
#points = [(1,1), (3,1), (5,1), (7,1), (9,1), (1,2), (3,2), (5,2), (7,2), (9,2)]

Sx = np.zeros((monomialLength,1))
Sy = np.zeros((monomialLength,1))
S = np.zeros((monomialLength, monomialLength))
Sxy = 0

vfEstimator = VectorFieldApproximator(2)



for pi in points:
	vi = sourceVF.sampleAtPoint(pi)
	vfEstimator.addMeasurement(pi,vi)
	wi = w(pi)
	Sx += vi[0]*wi
	Sy += vi[1]*wi
	S += np.dot(wi, wi.transpose())
	Sxy += vi[0]**2 + vi[1]**2#np.linalg.norm(vi)**2


k = 0
deltaMat = np.identity(S.shape[0])
# Bulk function calls to test VectorField functionality
flowVectors = sourceVF.sampleAtPoints(points)
wVectors = map(w, points)
scatterMats = map(lambda wVec: np.dot(wVec, wVec.transpose()), wVectors)


print(S)
print(Sx)
print(Sy)
print(Sxy)

# Commented code below fails with singular matrices
#a = np.linalg.solve(S,Sx)
#b = np.linalg.solve(S,Sy)

# Pseudoinverse to get around singular matrix failures
ridgeMat = S + k * deltaMat
pseudoInvS = np.linalg.pinv(ridgeMat)
a = np.dot(pseudoInvS,Sx)
b = np.dot(pseudoInvS,Sy)


print(a)
print(b)

error = np.dot(a.transpose(),np.dot(S,a)) + np.dot(b.transpose(), np.dot(S,b))
- 2 * np.dot(a.transpose(),Sx) - 2 * np.dot(b.transpose(),Sy) + Sxy

print(error)

eVals, eVecs = LA.eig(S)
index = np.argmin(eVals)
#print(eVals[index])
#print(eVecs[index])

approxVF = vfEstimator.approximate()

#approxVF = VectorField(lambda x,y: (np.dot(w((x,y)).transpose(), a)[0][0], np.dot(w((x,y)).transpose(), b)[0][0]))

ax2 = fig.add_subplot(312)
approxVF.quiver(ax2, grid)
ax2.axis([-2 * xCell, xDist + 2 * xCell, -2 * yCell, yDist + 2 * yCell])

fig = plt.gcf()
plt.show()