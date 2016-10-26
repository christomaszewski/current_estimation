import matplotlib.pyplot as plt
import numpy as np
import dill

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.vf.extents as vf_extents
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.sim as vf_sim
import LSPIV_toolkit.approx as vf_approx
import LSPIV_toolkit.core.plotting as vf_plot


xGrid = 25 #cells
yGrid = 10 #cells
xDist = 100 #meters
yDist = 50 #meters

displayGrid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

with open('../scenarios/pylon.scenario', mode='rb') as f:
	compoundVF = dill.load(f)

fieldView = vf_plot.SimpleFieldView(compoundVF, displayGrid, 1.0)

fieldView.quiver()

# Simulation

seedParticles = [(0, (5, 20)), (0, (15, 20)), (0, (25, 20)), (0, (35, 20)), (0, (45, 20)),
				(0, (55, 20)), (0, (65, 20)), (0, (75, 20)), (0, (85, 20)), (0, (95, 20))]


simulator = vf_sim.simulators.ParticleSimulator(compoundVF)

tracks = simulator.simulate(seedParticles, time=7, timestep=0.3)

vfEstimator = vf_approx.gp.CoregionalizedGPApproximator()
for track in tracks:
	vfEstimator.addMeasurements(track.getMeasurements(scoring='time'))

approxVF = vfEstimator.approximate(compoundVF.extents)

xSamples, ySamples = compoundVF.sampleGrid(displayGrid)
xgrid, ygrid = displayGrid.mgrid
magnitude = np.sqrt(xSamples**2 + ySamples**2)


fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('Source Field (Ground Truth)')
q1 = ax.quiver(xgrid, ygrid, xSamples, ySamples, magnitude, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)
ax.axis(compoundVF.plotExtents)
ax.hold(True)
fig.colorbar(q1, ax=ax)

c = 0
colors = ['red', 'blue', 'cyan', 'orange', 'green', 'black', 'yellow', 'purple', 'brown']
for track in tracks:
	t = np.asarray(track.getPointSequence())
	ax.scatter(t[:,0], t[:,1], c=colors[c])
	c = (c+1) % len(colors)


xSamples, ySamples = approxVF.sampleGrid(displayGrid)

magnitude = np.sqrt(xSamples**2 + ySamples**2)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('GP Approximation')
# fix colobar bounds when field differences are small
q2 = ax2.quiver(xgrid, ygrid, xSamples, ySamples, magnitude, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)
ax2.axis(approxVF.plotExtents)
fig.colorbar(q2, ax=ax2)

plt.show()
"""
plt.figure()
varx, vary = approxVF.sampleVarGrid(displayGrid)
print(varx)
plt.pcolormesh(xgrid, ygrid, vary, cmap=plt.cm.jet)
plt.show()"""

plt.pause(100)