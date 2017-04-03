import matplotlib.pyplot as plt
import numpy as np
import dill

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.sim as vf_sim
import LSPIV_toolkit.approx as vf_approx
import LSPIV_toolkit.core.plotting as vf_plot


plt.ion()

# Load Scenario
scenarioName = 'pylon'

# Scenario Field file name
scenarioFile = '../scenarios/' + scenarioName + '.scenario'

with open(scenarioFile, mode='rb') as f:
	compoundVF = dill.load(f)

xDist = compoundVF.extents.xRange[1] - compoundVF.extents.xRange[0]
yDist = compoundVF.extents.yRange[1] - compoundVF.extents.yRange[0]

xGrid = 35 #cells
yGrid = 15 #cells

grid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

# Initial Plot
sourceFieldView = vf_plot.SimpleFieldView(compoundVF, grid)
sourceFieldView.quiver()

# Sample source at random points
numPoints = 100

points = [tuple([np.random.rand(1)[0]*xDist, np.random.rand(1)[0]*yDist]) for _ in np.arange(numPoints)]

measurements = list(compoundVF.measureAtPoints(points))


# Test Reconstruction 
vfEstimator = vf_approx.gp.GPApproximator()

vfEstimator.addMeasurements(measurements)
approxVF = vfEstimator.approximate(compoundVF.extents)


# Plot reconstruction
approxFieldView = vf_plot.SimpleFieldView(approxVF, grid)
approxFieldView.quiver()
#sourceFieldView.plotPoints(points, 'black', 'o', 'Measurements')

sourceFieldView.save('../output/source.png')
approxFieldView.save('../output/approx.png')

fieldEval = vf_approx.eval.GridSampleComparison(grid, sourceField=compoundVF, approxField=approxVF)
print(fieldEval.meanError)
print(fieldEval.maxError)
print(fieldEval.minError)
fieldEval.plotErrors()
fieldEval.save('../output/reconstructionError.png')
	
