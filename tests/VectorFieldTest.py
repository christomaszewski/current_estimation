import matplotlib.pyplot as plt
import numpy as np

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.utils as vf_utils
import LSPIV_toolkit.core.plotting as vf_plot

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

vfPlot = vf_plot.SimpleFieldView(compoundVF, grid, 2)

vfPlot.quiver()

vfPlot2 = vf_plot.SimpleFieldView(sourceVF1, grid, 1)
vfPlot2.quiver()