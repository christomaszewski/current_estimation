import numpy as np
import matplotlib.pyplot as plt
import vf_utils.vector_field as vf
import vf_utils.approximate as vf_approx
import vf_utils.core as vf_core
import vf_utils.data_viz as vf_viz

import simulation.simulator as sim

vMax = 3 #m/s
riverWidth = 100 #meters

vfBounds = [0, riverWidth, 0, riverWidth]
sourceVF = vf.DevelopedPipeFlowField(riverWidth, vMax, vfBounds)

xGrid = 25 #cells
yGrid = 8 #cells
xDist = 100 #meters
yDist = 100 #meters

sampleGrid = vf_core.SampleGrid(xDist, yDist, xGrid, yGrid)

seedParticles = [(20,20), (50,50), (70,70)]
particleSim = sim.ParticleSimulator(sourceVF)

tracks = particleSim.simulate(seedParticles, time=2, timestep=0.5)

vfEstimator = vf_approx.PolynomialLSApproxmiator(2)

for track in tracks:
	vfEstimator.addMeasurements(track.getMeasurements())

approxVF = vfEstimator.approximate()
approxVF.setValidBounds(vfBounds)

fieldView = vf_viz.VectorFieldView(sampleGrid)
fieldView.addField(sourceVF)
fieldView.addField(approxVF)

c = 0
colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black']
for track in tracks:
	fieldView.plotTrack(track, colors[c])
	c+=1

fieldView.showPlots()