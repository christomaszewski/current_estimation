import numpy as np
import matplotlib.pyplot as plt
import vf_utils.vector_field as vf
import vf_utils.approximate as vf_approx
import vf_utils.core as vf_core
import vf_utils.data_viz as vf_viz
import vf_utils.analysis as vf_analysis

import simulation.simulator as sim

import time
import os

plt.ion()

# Scenario output folder
folderName = "scenario1B"

if not os.path.exists(folderName):
	os.makedirs(folderName)

# Generate unique run folder
subFolder =  folderName +"\\" + time.strftime('%d_%m_%Y_%H_%M_%S')

if not os.path.exists(subFolder):
	os.makedirs(subFolder)

vMax = 3 #m/s
riverWidth = 100 #meters

vfBounds = [0, riverWidth, 0, riverWidth]
sourceVF = vf.DevelopedPipeFlowField(riverWidth, vMax, vfBounds)

xGrid = 25 #cells
yGrid = 25 #cells
xDist = 100 #meters
yDist = 100 #meters

sampleGrid = vf_core.SampleGrid(xDist, yDist, xGrid, yGrid)

fieldView = vf_viz.VectorFieldView(sampleGrid)

vfEstimator = vf_approx.PolynomialLSApproxmiator(2)

measurementAnalysis  = vf_analysis.MeasurementProcessor(xDist, yDist, xGrid, yGrid)

sourceSim = sim.ParticleSimulator(sourceVF)

#seed particles are a defined as observations (time observed, (x,y))
seedParticles = [(0, (10, 30)), (5, (5,0))]

simTime = 10 #seconds
simTimeStep = 0.033 #30fps

renderTimeStep = 1 # how often should result images be generated

sourceFieldView = vf_viz.VectorFieldView(sampleGrid)
sourceFieldView.addField(sourceVF)
approxVF = None
particleTracks = []

for t in np.arange(renderTimeStep, simTime, renderTimeStep):
	particleTracks = sourceSim.simulate(seedParticles, t, simTimeStep)

	measurementAnalysis.clearMeasurements()

	for track in particleTracks:
		measurementAnalysis.addMeasurements(track.getMeasurements(scoring='time'))

	vfEstimator.clearMeasurements() 
	vfEstimator.addMeasurements(measurementAnalysis.getMeasurements())

	approxVF = vfEstimator.approximate(vfBounds)
	#approxVF.setValidBounds(vfBounds)

	approxFileName = subFolder + "\\reconstruction_" + str(t) + ".png"

	fieldView.clearFields()
	if (approxVF is not None):
		fieldView.addField(approxVF)

	fieldView.saveFig(approxFileName, t)
	
	sourceFieldView.plot()
	c = 0
	colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black']
	for track in particleTracks:
		sourceFieldView.plotTrack(track, colors[c])
		c+=1
	
	trackFileName = subFolder + "\\track_" + str(t) + ".png"
	sourceFieldView.saveFig(trackFileName, t)

	#time.sleep(1)

	measurementFileName = subFolder + "\\measurements_" + str(t) + ".png"

	#fieldView.closePlots()
	measurementAnalysis.drawMeasurementGrid()
	measurementAnalysis.saveFig(measurementFileName, t)

boatParticle = [(10, (90, 0))]

simTime = 15

for t in np.arange(10, simTime, renderTimeStep):
	boatTrack = sourceSim.simulate(boatParticle, t, simTimeStep)

	measurementAnalysis.clearMeasurements()

	tracks = []
	tracks.extend(particleTracks)
	tracks.extend(boatTrack)

	for track in tracks:
		measurementAnalysis.addMeasurements(track.getMeasurements(scoring='time'))

	vfEstimator.clearMeasurements() 
	vfEstimator.addMeasurements(measurementAnalysis.getMeasurements())

	approxVF = vfEstimator.approximate(vfBounds)
	#approxVF.setValidBounds(vfBounds)

	approxFileName = subFolder + "\\reconstruction_" + str(t) + ".png"

	fieldView.clearFields()
	if (approxVF is not None):
		fieldView.addField(approxVF)

	fieldView.saveFig(approxFileName, t)
	
	sourceFieldView.plot()
	c = 0
	colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black']
	for track in tracks:
		sourceFieldView.plotTrack(track, colors[c])
		c+=1
	
	trackFileName = subFolder + "\\track_" + str(t) + ".png"
	sourceFieldView.saveFig(trackFileName, t)

	#time.sleep(1)

	measurementFileName = subFolder + "\\measurements_" + str(t) + ".png"

	#fieldView.closePlots()
	measurementAnalysis.drawMeasurementGrid()
	measurementAnalysis.saveFig(measurementFileName, t)





streamLineParticles = [(0, (5, 0)), (0, (15, 0)), (0, (25, 0)), (0, (35, 0)), (0, (45, 0)),
						(0, (55, 0)), (0, (65, 0)), (0, (75, 0)), (0, (85, 0)), (0, (95, 0))]

approxSim = sim.ParticleSimulator(approxVF)

sourceStreamLines = sourceSim.simulate(streamLineParticles, 15, simTimeStep)
approxStreamLines = approxSim.simulate(streamLineParticles, 15, simTimeStep)


sourceFieldView.plot()
fieldView.plot()



c = 0
colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black', 'yellow', 'purple', 'brown']
for track in sourceStreamLines:
	sourceFieldView.plotTrack(track, colors[c])
	c = (c+1) % len(colors)

c = 0
for track in approxStreamLines:
	fieldView.plotTrack(track, colors[c])
	c = (c+1) % len(colors)

streamLineFileName = subFolder + "\\" + folderName + "_streamline_source.png"
sourceFieldView.saveFig(streamLineFileName, 15)

streamLineFileName = subFolder + "\\" + folderName + "_streamline_approx.png"
fieldView.saveFig(streamLineFileName, 15)