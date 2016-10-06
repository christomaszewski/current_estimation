import numpy as np
import matplotlib.pyplot as plt
import vf_utils.vector_field as vf
import vf_utils.approximate as vf_approx
import vf_utils.core as vf_core
import vf_utils.data_viz as vf_viz
import simulation.simulator as sim
from scipy.interpolate import griddata

vMax = 3 #m/s
xGrid = 25 #cells
yGrid = 8 #cells
xDist = 100 #meters
yDist = 100 #meters

riverWidth = 100 #meters

vfFunc = lambda x, y: (0, (4 * x / riverWidth - 4 * x**2 / riverWidth**2) * vMax)
vfBounds = [-10, riverWidth+10, -10, riverWidth+10]
sourceVF = vf.VectorField(vfFunc, vfBounds)

uniformVF = vf.UniformVectorField((0,1), vfBounds)

grid = vf_core.SampleGrid(xDist, yDist, xGrid, yGrid)

fieldView = vf_viz.VectorFieldView(grid)
fieldView.addField(sourceVF)

gridX, gridY = grid.mgrid

points = [(x, yDist / 2) for x in range(2, 100, 4)]
#points = [(50, 50)]
#print(np.asarray(points))
#print(list(sourceVF.sampleAtPoints(points)))
#print(np.asarray(list(sourceVF.sampleAtPoints(points)))[:, 1])
print(np.asarray(points))
print(np.asarray(list(sourceVF.sampleAtPoints(points)))[:, 1])
interpPriorY = griddata(np.asarray(points), np.asarray(list(sourceVF.sampleAtPoints(points)))[:, 1], (gridX, gridY), method="nearest")
interpPriorX = griddata(np.asarray(points), np.asarray(list(sourceVF.sampleAtPoints(points)))[:, 0], (gridX, gridY), method="nearest")

flatGridX = gridX.flatten()
flatGridY = gridY.flatten()
flatInterpX = interpPriorX.flatten()
flatInterpY = interpPriorY.flatten()
numElements = len(flatGridY)
measurementPrior = []

for i in np.arange(0, numElements, 1):
	point = (flatGridX[i], flatGridY[i])
	vector = (flatInterpX[i], flatInterpY[i])
	#print(vector)
	measurementPrior.append((point, vector))


#points = [(3,3), (5,8), (9,1)]
#points = [(1,1), (3,1), (1,5), (3,5)]
#points = [(1, 1), (3, 1), (5, 1), (7, 1), (9, 1)]
#points = [(1, 1), (3, 1), (5, 1), (7, 1), (9, 1), (1, 2), (3, 2), (5, 2), (7, 2), (9, 2)]


vfEstimator = vf_approx.PolynomialLSApproxmiator(2)

for pi in points:
	vfEstimator.addMeasurement(sourceVF.measureAtPoint(pi))

approxVF = vfEstimator.approximate()
approxVF.setValidBounds(vfBounds)

fieldView.addField(approxVF)

vfEstimator.clearMeasurements()

#measurementPrior = uniformVF.generateMeasurementsOnGrid(grid)


for m in measurementPrior:
	vfEstimator.addMeasurement(m)

for pi in points:
	vfEstimator.addMeasurement(sourceVF.measureAtPoint(pi))

approxVF2 = vfEstimator.approximate()
approxVF2.setValidBounds(vfBounds)

fieldView.addField(approxVF2)

#fieldView.showPlots()

seedParticles = [(20,20), (50,50), (70,70)]

simulator = sim.Simulator(sourceVF)
tracks = simulator.simulate(seedParticles, 2, 0.5)

print(np.asarray(tracks[0])[:,1])

trackParser = vf_core.TrackParser()
measurements = trackParser.tracksToMeasurements(tracks, 0.5)

print(measurements)

vfEstimator.clearMeasurements()
vfEstimator.addMeasurements(measurements)

approxVF3 = vfEstimator.approximate()
approxVF3.setValidBounds(vfBounds)

fieldView.addField(approxVF3)

c = 0
colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black']
for track in tracks:
	fieldView.plotTrack(track, colors[c])
	c+=1

fieldView.showPlots()