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


xGrid = 20 #cells
yGrid = 10 #cells
xDist = 100 #meters
yDist = 50 #meters

displayGrid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

with open('../scenarios/twin_channel.scenario', mode='rb') as f:
	compoundVF = dill.load(f)

fieldView = vf_plot.SimpleFieldView(compoundVF, displayGrid, 1.0)

fieldView.quiver()

# Simulation

seedParticles = [(0, (5, 20)), (3, (20, 5)), (6, (35, 35)), (5, (80, 25)), (6, (45, 15)),
					(7, (55, 5)), (7, (60, 30)), (8, (93, 5))]


simulator = vf_sim.simulators.ParticleSimulator(compoundVF)

tracks = simulator.simulate(seedParticles, time=11, timestep=0.33)

vfEstimator = vf_approx.gp.GPApproximator()
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