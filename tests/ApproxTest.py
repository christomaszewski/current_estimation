import matplotlib.pyplot as plt
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
scenarioName = 'pylon'

# Scenario Field file name
scenarioFile = '../scenarios/' + scenarioName + '.scenario'

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

vfEstimator = vf_approx.gp.GPApproximator()

measurements = compoundVF.generateMeasurementsOnGrid(displayGrid)
print(measurements)
vfEstimator.addMeasurements(measurements)

approxVF = vfEstimator.approximate(compoundVF.extents)

evaluator.changeFields(sourceField=compoundVF, approxField=approxVF)
streamEval.changeFields(sourceField=compoundVF, approxField=approxVF)

print("Error: ", evaluator.normalError)
print("Stream Error: ", streamEval.normalError)


approxFieldView.changeField(approxVF)
approxFieldView.quiver()


plt.pause(100)