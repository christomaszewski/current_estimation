import matplotlib.pyplot as plt
import numpy as np

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

# Domain Extents [0, 100, 0, 50]
domainExtents = vf_extents.FieldExtents((0.0, xDist), (0.0, yDist))

# Define a uniform flow across the domain
uniformFlow = field_lib.UniformVectorField(flowVector=(-0.2, 0.5), fieldExtents=domainExtents)

# Simulate bridge pylon with two structured flows on either side of a region of slow flow

# Left channel [0, 30]
lcWidth = 60.0
lcVMax = 3.0
lcExtents = vf_extents.FieldExtents((0.0, lcWidth), (0.0, yDist))
lcFlow = field_lib.DevelopedPipeFlowField(lcWidth, lcVMax, lcExtents)

# Center chanel [60, 70]
cExtentsPrePylon = vf_extents.FieldExtents((lcWidth, lcWidth+10), (0.0, yDist/2.0))
cExtentsPostPylon = vf_extents.FieldExtents((lcWidth, lcWidth+10), (yDist/2.0, yDist))
prePylonFlow = field_lib.UniformVectorField(flowVector=(0.0, 1.0), fieldExtents=cExtentsPrePylon)
postPylonFlow = field_lib.UniformVectorField(flowVector=(0.0, 0.5), fieldExtents=cExtentsPostPylon)
divFlow = field_lib.DivergingFlowField(1.0, (65, 0), cExtentsPrePylon)
convFlow = field_lib.ConvergingFlowField(1.0, (65, 0), cExtentsPostPylon)
centerFlow = field_lib.CompoundVectorField(prePylonFlow, postPylonFlow, divFlow, convFlow)
# Right channel [70, 100]
rcWidth = 30.0
rcVMax = 1.5
rcExtents = vf_extents.FieldExtents((70.0, 100.0), (0.0, yDist))
rcFlow = field_lib.DevelopedPipeFlowField(rcWidth, rcVMax, rcExtents, offset=(70.0, 0.0))


compoundVF = field_lib.CompoundVectorField(uniformFlow, lcFlow, centerFlow, rcFlow)

displayGrid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

fieldView = vf_plot.SimpleFieldView(compoundVF, displayGrid, 1.0)

fieldView.quiver()

# Simulation

seedParticles = [(0, (5, 20)), (0, (15, 20)), (0, (25, 20)), (0, (35, 20)), (0, (45, 20)),
				(0, (55, 20)), (0, (65, 20)), (0, (75, 20)), (0, (85, 20)), (0, (95, 20))]


simulator = vf_sim.simulators.ParticleSimulator(compoundVF)

tracks = simulator.simulate(seedParticles, time=7, timestep=0.3)

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

plt.pause(100)