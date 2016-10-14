import matplotlib.pyplot as plt
import numpy as np

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.sim as vf_sim
import LSPIV_toolkit.approx as vf_approx

plt.ion()

# Scenario Setup

vMax = 3 #m/s
riverWidth = 100 #meters

sourceVF = field_lib.DevelopedPipeFlowField(riverWidth, vMax)

xGrid = 25 #cells
yGrid = 8 #cells
xDist = 100 #meters
yDist = 100 #meters

grid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

# Initial Plot

xSamples, ySamples = sourceVF.sampleGrid(grid)

magnitude = np.sqrt(xSamples**2 + ySamples**2)

xgrid, ygrid = grid.mgrid
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.quiver(xgrid, ygrid, xSamples, ySamples, magnitude, cmap=plt.cm.jet)
ax.axis(sourceVF.plotExtents)
ax.hold(True)

# Simulation

"""seedParticles = [(0, (5, 5)), (0, (15, 5)), (0, (25, 5)), (0, (35, 5)), (0, (45, 5)),
				(0, (55, 5)), (0, (65, 5)), (0, (75, 5)), (0, (85, 5)), (0, (95, 5))]
"""
seedParticles = [(0, (10, 30)), (0, (5,5))]

simulator = vf_sim.simulators.ParticleSimulator(sourceVF)

tracks = simulator.simulate(seedParticles, time=2, timestep=0.5)

# Track Plotting

c = 0
colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black', 'yellow', 'purple', 'brown']
for track in tracks:
	t = np.asarray(track.getPointSequence())
	ax.scatter(t[:,0], t[:,1], c=colors[c])
	c = (c+1) % len(colors)

plt.show()

# Approximation 

vfEstimator = vf_approx.polynomial.PolynomialLSApproxmiator(polyDegree=2)

for track in tracks:
	vfEstimator.addMeasurements(track.getMeasurements(scoring='time'))

approxVF = vfEstimator.approximate(sourceVF.extents)

# Approximation Plotting

xSamples, ySamples = approxVF.sampleGrid(grid)

magnitude = np.sqrt(xSamples**2 + ySamples**2)

ax2 = fig.add_subplot(1,2,2)
ax2.quiver(xgrid, ygrid, xSamples, ySamples, magnitude, cmap=plt.cm.jet)
ax2.axis(approxVF.plotExtents)

plt.show()

fig.savefig("PolyApproxTest.png", bbox_inches='tight')

