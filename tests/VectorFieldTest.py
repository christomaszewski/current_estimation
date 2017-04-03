import matplotlib.pyplot as plt
import numpy as np
import dill

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.core.plotting as vf_plot

# Scenario Name
scenarioName = 'pylon'

# Scenario Field file name
scenarioFile = '../scenarios/' + scenarioName + '.scenario'

with open(scenarioFile, mode='rb') as f:
	compoundVF = dill.load(f)

xGrid = 25 #cells
yGrid = 15 #cells

xDist = compoundVF.extents.xRange[1]
yDist = compoundVF.extents.yRange[1]

grid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

vfPlot = vf_plot.SimpleFieldView(compoundVF, grid, 5)
vfPlot.setTitle('LSPIV Approximation')
vfPlot.quiver()

vfPlot.save('../output/' + scenarioName + '.png')
