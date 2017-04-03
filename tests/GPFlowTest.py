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

# Scenario Name
scenarioName = 'pylon'

# Scenario Field file name
scenarioFile = '../scenarios/' + scenarioName + '.scenario'

with open(scenarioFile, mode='rb') as f:
	compoundVF = dill.load(f)

xDist = compoundVF.extents.xRange[1]
yDist = compoundVF.extents.yRange[1]

xGrid = 35 #cells
yGrid = 15 #cells

sampleGrid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

sourceFieldView = vf_plot.SimpleFieldView(compoundVF, sampleGrid)
sourceFieldView.setTitle('Sampled Source Field')

sourceFieldView.quiver()

approxFieldView = vf_plot.SimpleFieldView(grid=sampleGrid)
approxFieldView.setTitle('GP Approximation')

vfEstimator = vf_approx.gpflow.GPFlowApproximator()

measurements = compoundVF.generateMeasurementsOnGrid(sampleGrid)
vfEstimator.addMeasurements(measurements)

approxVF = vfEstimator.approximate(compoundVF.extents)

approxFieldView.changeField(approxVF)
approxFieldView.quiver()

sourceFieldView.save('../output/source.png')
approxFieldView.save('../output/approx.png')