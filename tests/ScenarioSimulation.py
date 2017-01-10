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
scenarioName = 'approx_pylon'

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

seedParticles = [(0, p) for p in displayGrid.cellCenters]

streamEval = vf_approx.eval.StreamLineComparison(seedParticles=seedParticles,
				sourceField=compoundVF, simTime=5, simRes=0.1)
evaluator = vf_approx.eval.GridSampleComparison(displayGrid, sourceField=compoundVF)
streamErrors = []
errors = []
vfEstimator = vf_approx.gp.GPApproximator()
measurementAnalysis  = vf_analysis.MeasurementProcessor(xDist, yDist, xGrid)

# Simulation

#seedParticles = [(2, (5, 20)), (0, (20, 3)), (4, (35, 12)), (3, (80, 10)), (6, (42, 15)),
#					(3, (64, 5)), (5, (59, 25)), (7, (96, 5))]

seedParticles = [(0, (5, 20)), (0, (20, 3)), (0, (40, 12)), (0, (57, 28)), (0, (64, 5))]

simTime = 11 #seconds
simTimeStep = 0.033 #30fps

boatTime = 10
boatParticle = [(simTime-1, (67,13))]
#(simTime-1, (66,15)), 

renderTimeStep = 1 # number of seconds in between approximations

totalTime = simTime + boatTime - 1

simulator = vf_sim.simulators.ParticleSimulator(compoundVF, noise=0)
particleTracks = []
for t in np.arange(renderTimeStep, simTime+boatTime, renderTimeStep):
	particleTracks = simulator.simulate(seedParticles, t, simTimeStep)
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
		#vfEstimator.addMeasurements(track.getMeasurements(scoring='time'))

	#print(measurementAnalysis.getMeasurements())

	for boat in boatTracks:
		tr = boat
		measurementAnalysis.addMeasurements(tr.getMeasurements(scoring='time'))
		sourceFieldView.plotTrack(tr, colors[0], marker='^')

	#vfEstimator.clearMeasurements()
	vfEstimator.addMeasurements(measurementAnalysis.getMeasurements())
	approxVF = vfEstimator.approximate(compoundVF.extents)

	streamEval.changeFields(approxField=approxVF)
	evaluator.changeFields(approxField=approxVF)
	print("Error: ", evaluator.normalError)
	errors.append(evaluator.normalError)
	streamErrors.append(streamEval.normalError)
	print("Streamline Error: ", streamEval.normalError)

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
"""

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

	streamEval.changeFields(approxField=approxVF)
	evaluator.changeFields(approxField=approxVF)
	print("Error: ", evaluator.normalError)
	errors.append(evaluator.normalError)
	streamErrors.append(streamEval.normalError)
	print("Streamline Error: ", streamEval.normalError)


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
"""

"""

boatTime = 10
boatParticle = [(simTime+boatTime-1, (99,5))]
#(simTime-1, (66,15)), 

renderTimeStep = 1 # number of seconds in between approximations


# Boat simulation
for t in np.arange(simTime+boatTime, simTime+2*boatTime, renderTimeStep):
	boatTracks2 = simulator.simulate(boatParticle, t, simTimeStep)

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

	for boat in boatTracks2:
		tr = boat
		measurementAnalysis.addMeasurements(tr.getMeasurements(scoring='time'))
		sourceFieldView.plotTrack(tr, colors[0], marker='^')

	#print(measurementAnalysis.getMeasurements())

	#vfEstimator.clearMeasurements()
	vfEstimator.addMeasurements(measurementAnalysis.getMeasurements())
	approxVF = vfEstimator.approximate(compoundVF.extents)

	streamEval.changeFields(approxField=approxVF)
	evaluator.changeFields(approxField=approxVF)
	print("Error: ", evaluator.normalError)
	errors.append(evaluator.normalError)
	streamErrors.append(streamEval.normalError)
	print("Streamline Error: ", streamEval.normalError)


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


"""

with open('../output/errors.error', mode='rb') as f:
		oldErrors = dill.load(f)

with open('../output/streamErrors.error', mode='rb') as f:
		oldStreamErrors = dill.load(f)

evaluator.plotErrors()
evaluator.save(subFolder + '/errorVectors.png')

errorArray = np.asarray(errors)
timeArray = np.arange(len(errorArray)) + 1
f = plt.figure()
ax = plt.subplot(211)
h1 = ax.plot(timeArray, errorArray[:, 0], color='blue', label='Augmented LSPIV')
ax.hold(True)
h2 = ax.plot(timeArray, oldErrors[:len(timeArray), 0], color='black', label='LSPIV')
ax.set_title("X Error")
ax.grid(True)
maxY = np.max(oldErrors[:,0])
ax.axis([0, totalTime, 0, maxY])
x = np.arange(10, 21)
#y = np.arange(20, 31)
handle = ax.fill_between(x, 0, maxY, facecolor='green', alpha=0.5, label='ASV Present')
#handle1 = ax.fill_between(y, 0, maxY, facecolor='yellow', alpha=0.5, label='ASV Drift 2')
ax.legend(loc=2)
#ax.legend(handles=[handle])
#ax.set_xlabel('Time (s)')
ax.set_ylabel('Error (m/s)')


ax2 = plt.subplot(212)
h3 = ax2.plot(timeArray, errorArray[:, 1], color='red', label='Augmented LSPIV')
ax2.hold(True)
h4 = ax2.plot(timeArray, oldErrors[:len(timeArray), 1], color='black', label='LSPIV')
ax2.set_title("Y Error")
ax2.grid(True)
maxY = np.max(oldErrors[:,1])
ax2.axis([0, totalTime, 0, maxY])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (m/s)')
handle2 = ax2.fill_between(x, 0, maxY, facecolor='green', alpha=0.5, label='ASV Present')
#handle3 = ax2.fill_between(y, 0, maxY, facecolor='yellow', alpha=0.5, label='ASV Drift 2')
ax2.legend(loc=3)
#ax2.legend(handles=[handle2])


plt.show()


errorFileName = subFolder + '/errors.png'
f.savefig(errorFileName, bbox_inches='tight')
plt.pause(10)

errorArray = np.asarray(streamErrors)
f = plt.figure()
ax = plt.subplot(211)
h1 = ax.plot(timeArray, errorArray[:, 0], color='blue', label='Augmented LSPIV')
ax.hold(True)
h2 = ax.plot(timeArray, oldStreamErrors[:len(timeArray), 0], color='black', label='LSPIV')
ax.set_title("X Streamline Error")
ax.grid(True)
ax.set_ylabel('Error (m)')
maxY = np.max(oldStreamErrors[:,0])
ax.axis([0, totalTime, 0, maxY])
handle = ax.fill_between(x, 0, maxY, facecolor='green', alpha=0.5, label='ASV Present')
#handle1 = ax.fill_between(y, 0, maxY, facecolor='yellow', alpha=0.5, label='ASV Drift 2')
ax.legend(loc=3)
#ax.legend(handles=[handle])

ax2 = plt.subplot(212)
h3 = ax2.plot(timeArray, errorArray[:, 1], color='red', label='Augmented LSPIV')
ax2.hold(True)
h4 = ax2.plot(timeArray, oldStreamErrors[:len(timeArray), 1], color='black', label='LSPIV')
ax2.set_title("Y Streamline Error")
ax2.grid(True)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (m)')
maxY = np.max(errorArray[:,1])
ax2.axis([0, totalTime, 0, maxY])

handle2 = ax2.fill_between(x, 0, maxY, facecolor='green', alpha=0.5, label='ASV Present')
#handle3 = ax2.fill_between(y, 0, maxY, facecolor='yellow', alpha=0.5, label='ASV Drift 2')
ax2.legend(loc=3)
#ax2.legend(handles=[handle2])

plt.show()
errorFileName = subFolder + '/StreamlineErrors.png'
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


