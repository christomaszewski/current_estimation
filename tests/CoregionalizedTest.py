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

sourceVF1 = field_lib.DevelopedPipeFlowField(channelWidth=50, vMax=vMax)
sourceVF2 = field_lib.DevelopedPipeFlowField(channelWidth=50, vMax=2*vMax, offset=(50,0))
compoundVF = field_lib.CompoundVectorField(sourceVF1, sourceVF2)


xGrid = 25 #cells
yGrid = 8 #cells
xDist = 100 #meters
yDist = 50 #meters

grid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

# Initial Plot

xSamples, ySamples = compoundVF.sampleGrid(grid)
xgrid, ygrid = grid.mgrid
magnitude = np.sqrt(xSamples**2 + ySamples**2)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('Source Field (Ground Truth)')
q1 = ax.quiver(xgrid, ygrid, xSamples, ySamples, magnitude, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)
ax.axis(compoundVF.plotExtents)
ax.hold(True)
fig.colorbar(q1, ax=ax)

# Simulation

seedParticles = [(0, (5, 5)), (0, (15, 5)), (0, (25, 5)), (0, (35, 5)), (0, (45, 5)),
				(0, (55, 5)), (0, (65, 5)), (0, (75, 5)), (0, (85, 5)), (0, (95, 5))]

#seedParticles = [(0, (10, 30)), (0, (5,5))]

simulator = vf_sim.simulators.ParticleSimulator(compoundVF)

tracks = simulator.simulate(seedParticles, time=2, timestep=0.5)

# Track Plotting

c = 0
colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black', 'yellow', 'purple', 'brown']
for track in tracks:
	t = np.asarray(track.getPointSequence())
	ax.scatter(t[:,0], t[:,1], c=colors[c])
	c = (c+1) % len(colors)

plt.show()


# GP Approximation

vfEstimator = vf_approx.gp.CoregionalizedGPApproximator()
for track in tracks:
	vfEstimator.addMeasurements(track.getMeasurements(scoring='time'))

approxVF = vfEstimator.approximate(compoundVF.extents)

# Approximation Plotting

xSamples, ySamples = approxVF.sampleGrid(grid)

magnitude = np.sqrt(xSamples**2 + ySamples**2)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('GP Approximation')

q2 = ax2.quiver(xgrid, ygrid, xSamples, ySamples, magnitude, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)
ax2.axis(approxVF.plotExtents)
fig.colorbar(q2, ax=ax2)
plt.show()

fig.savefig("CoregionalizedTest.png", bbox_inches='tight')