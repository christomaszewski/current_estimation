import cv2
import numpy as np

from context import vf_utils
import vf_utils.approx as vf_approx
import vf_utils.core as vf_core
import vf_utils.vector_field as vf


vMax = 3 #m/s
riverWidth = 100 #meters

vfBounds = [0, riverWidth, 0, riverWidth]
sourceVF = vf.DevelopedPipeFlowField(riverWidth, vMax, vfBounds)

xGrid = 25 #cells
yGrid = 8 #cells
xDist = 100 #meters
yDist = 100 #meters

samplePoints = [(20,50), (10, 50), (30,50), (60, 50), (50,50), (70,50), (80, 50)]
samplePoints.extend([(20,60), (10, 60), (30,60), (60, 60), (50,60), (70,60), (80, 60)])
measurements = [vf_core.Measurement(p,sourceVF.sampleAtPoint(p)) for p in samplePoints]
print(measurements)

vfApproximator = vf_approx.gp.GPApproximator()

vfApproximator.addMeasurements(measurements)

modelX, modelY = vfApproximator.approximate()

testPoints = [(10, 50)]
testX = np.asarray(testPoints)

muX, varX = modelX.predict(testX, kern=vfApproximator._K, full_cov=True)
muY, varY = modelY.predict(testX, kern=vfApproximator._K, full_cov=True)

for velX, velY in zip(muX, muY):
	print(velX, velY)

""" Coregionalized Stuff
outputIndex = np.ones((len(testPoints), 1))

modelInput = np.hstack([testX, outputIndex])
#testX = np.reshape(testX, (2,2))

muY, varY = model.predict_noiseless(modelInput)

print(muY)

outputIndex *= 0

modelInput = np.hstack([testX, outputIndex])

muX, varX = model.predict(modelInput, include_likelihood=False)

print(muX)"""