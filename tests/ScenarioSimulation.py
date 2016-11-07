import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dill
import os
import time
import glob
import imageio

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.vf.extents as vf_extents
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.analysis as vf_analysis
import LSPIV_toolkit.sim as vf_sim
import LSPIV_toolkit.approx as vf_approx
import LSPIV_toolkit.core.plotting as vf_plot


plt.ion()

# Scenario Name
scenarioName = 'twin_channel'

# Scenario Field file name
scenarioFile = '../scenarios/' + scenarioName + '.scenario'

# Scenario output folder
folderName = '../output/' + scenarioName

if not os.path.exists(folderName):
	os.makedirs(folderName)

# Generate unique run folder
subFolder =  folderName + '/' + time.strftime('%d_%m_%Y_%H_%M_%S')

if not os.path.exists(subFolder):
	os.makedirs(subFolder)

with open(scenarioFile, mode='rb') as f:
	compoundVF = dill.load(f)

xDist = compoundVF.extents.xRange[1]
yDist = compoundVF.extents.yRange[1]

xGrid = 20 #cells
yGrid = 10 #cells

displayGrid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

sourceFieldView = vf_plot.SimpleFieldView(compoundVF, displayGrid)
sourceFieldView.setTitle('Source Field (Ground Truth)')

sourceFieldView.quiver()

approxFieldView = vf_plot.SimpleFieldView(grid=displayGrid)
approxFieldView.setTitle('GP Approximation')
approxFieldView.setClim(sourceFieldView.clim)

evaluator = vf_approx.eval.GridSampleComparison(displayGrid, sourceField=compoundVF)
errors = []
vfEstimator = vf_approx.gp.GPApproximator()
measurementAnalysis  = vf_analysis.MeasurementProcessor(xDist, yDist, xGrid)

# Simulation

seedParticles = [(2, (5, 20)), (0, (20, 3)), (4, (35, 12)), (3, (80, 10)), (6, (42, 15)),
					(3, (64, 5)), (5, (59, 25)), (7, (96, 5))]

simTime = 11 #seconds
simTimeStep = 0.033 #30fps

boatTime = 4
boatParticle = [(simTime-1, (66,15)), (simTime-1, (74,31))]
#(simTime-1, (66,15)), 

renderTimeStep = 1 # number of seconds in between approximations

simulator = vf_sim.simulators.ParticleSimulator(compoundVF)
particleTracks = []
for t in np.arange(renderTimeStep, simTime, renderTimeStep):
	particleTracks = simulator.simulate(seedParticles, t, simTimeStep)
	
	measurementAnalysis.clearMeasurements()

	sourceFieldView.setAnnotation("Time: " + str(t))
	approxFieldView.setAnnotation("Time: " + str(t))

	sourceFieldView.quiver()

	vfEstimator.clearMeasurements()
	c = 0
	colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black', 'yellow', 'purple', 'brown']
	for track in particleTracks:
		measurementAnalysis.addMeasurements(track.getMeasurements(scoring='time'))
		sourceFieldView.plotTrack(track, colors[c])
		c = (c+1) % len(colors)
		#vfEstimator.addMeasurements(track.getMeasurements(scoring='time'))

	#print(measurementAnalysis.getMeasurements())

	#vfEstimator.clearMeasurements()
	vfEstimator.addMeasurements(measurementAnalysis.getMeasurements())
	approxVF = vfEstimator.approximate(compoundVF.extents)

	evaluator.changeFields(approxField=approxVF)
	print("Error: ", evaluator.normalError)
	errors.append(evaluator.normalError)

	approxFieldView.changeField(approxVF)
	approxFieldView.quiver()
	measurementAnalysis.drawMeasurementGrid()

	#Save images
	sourceFileName = subFolder + '/source_' + str(t) + '.png'
	approxFileName = subFolder + '/approx_' + str(t) + '.png'
	measurementFileName = subFolder + '/measurement_' + str(t) + '.png'

	sourceFieldView.save(sourceFileName)
	approxFieldView.save(approxFileName)
	measurementAnalysis.saveFig(measurementFileName, t)


# Boat simulation
for t in np.arange(simTime, simTime+boatTime, renderTimeStep):
	boatTracks = simulator.simulate(boatParticle, t, simTimeStep)

	measurementAnalysis.clearMeasurements()

	sourceFieldView.setAnnotation("Time: " + str(t))
	approxFieldView.setAnnotation("Time: " + str(t))

	sourceFieldView.quiver()

	vfEstimator.clearMeasurements()

	c = 0
	colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black', 'yellow', 'purple', 'brown']
	for track in particleTracks:
		measurementAnalysis.addMeasurements(track.getMeasurements(scoring='time'))
		sourceFieldView.plotTrack(track, colors[c])
		c = (c+1) % len(colors)

	# Draw Boat
	for boat in boatTracks:
		tr = boat
		measurementAnalysis.addMeasurements(tr.getMeasurements(scoring='time'))
		sourceFieldView.plotTrack(tr, colors[0], marker='^')


	#print(measurementAnalysis.getMeasurements())

	#vfEstimator.clearMeasurements()
	vfEstimator.addMeasurements(measurementAnalysis.getMeasurements())
	approxVF = vfEstimator.approximate(compoundVF.extents)

	evaluator.changeFields(approxField=approxVF)
	print("Error: ", evaluator.normalError)
	errors.append(evaluator.normalError)

	approxFieldView.changeField(approxVF)
	approxFieldView.quiver()
	measurementAnalysis.drawMeasurementGrid()

	#Save images
	sourceFileName = subFolder + '/source_' + str(t) + '.png'
	approxFileName = subFolder + '/approx_' + str(t) + '.png'
	measurementFileName = subFolder + '/measurement_' + str(t) + '.png'

	sourceFieldView.save(sourceFileName)
	approxFieldView.save(approxFileName)
	measurementAnalysis.saveFig(measurementFileName, t)


errorArray = np.asarray(errors)
timeArray = np.arange(len(errorArray)) + 1
f = plt.figure()
plt.subplot(211)
plt.plot(timeArray, errorArray[:, 0], color='blue')
plt.title("X Error")
plt.grid(True)

plt.subplot(212)
plt.plot(timeArray, errorArray[:, 1], color='red')
plt.title("Y Error")
plt.grid(True)
plt.show()
errorFileName = subFolder + '/errors.png'
f.savefig(errorFileName, bbox_inches='tight')
plt.pause(10)
# Gif generation

measurementGlob = sorted(glob.glob(subFolder + '/measurement_*.png'), key=os.path.getmtime)
sourceGlob = sorted(glob.glob(subFolder + '/source_*.png'), key=os.path.getmtime)
approxGlob = sorted(glob.glob(subFolder + '/approx_*.png'), key=os.path.getmtime)

images = []

for file in measurementGlob:
	images.append(imageio.imread(file))

filename = subFolder + '/measurement.gif'

imageio.mimsave(filename, images, duration=1, loop=2)

filename = subFolder + '/source.gif'

images = []

for file in sourceGlob:
	images.append(imageio.imread(file))

imageio.mimsave(filename, images, duration=1, loop=2)

filename = subFolder + '/approx.gif'

images = []

for file in approxGlob:
	images.append(imageio.imread(file))

imageio.mimsave(filename, images, duration=1, loop=2)


