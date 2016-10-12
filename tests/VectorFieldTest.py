import matplotlib.pyplot as plt
import numpy as np

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.utils as vf_utils

vMax = 3 #m/s
riverWidth = 100 #meters

sourceVF = field_lib.DevelopedPipeFlowField(riverWidth, vMax)

xGrid = 25 #cells
yGrid = 8 #cells
xDist = 100 #meters
yDist = 100 #meters

grid = vf_utils.SampleGrid(xDist, yDist, xGrid, yGrid)

xSamples, ySamples = sourceVF.sampleGrid(grid)

magnitude = np.sqrt(xSamples**2 + ySamples**2)

xgrid, ygrid = grid.mgrid
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.quiver(xgrid, ygrid, xSamples, ySamples, magnitude, cmap=plt.cm.jet)
ax.axis(sourceVF.plotExtents)

plt.show()