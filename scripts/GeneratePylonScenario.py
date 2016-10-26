import matplotlib.pyplot as plt
import numpy as np
import dill

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.vf.extents as vf_extents

""" Script to generate a flow simulating a bridge pylon with two structured 
	flows on either side of a region of slow flow, which divereges and then 
	converges around the simulated pylon location
"""

# Scenario Output filename
fileName = "../scenarios/pylon.scenario"


# Display Grid Definition
xGrid = 50 #cells
yGrid = 10 #cells
xDist = 100 #meters
yDist = 50 #meters

# Domain Extents [0, 100, 0, 50]
domainExtents = vf_extents.FieldExtents((0.0, xDist), (0.0, yDist))

# Define channels making up the flow
lcWidth = 60.0
ccWidth = 10.0
rcWidth = xDist - lcWidth - ccWidth

# Compute partition axes (these should be sorted)
partitionAxes = (lcWidth, lcWidth + ccWidth)

# Parition domain extents into channel extents
lcExtents, ccExtents, rcExtents = domainExtents.xSplit(*partitionAxes)

# Define a pylon location (for now just y position in center channel)
pylonYPos = yDist / 2.0
pylonXPos = lcWidth + ccWidth / 2.0

# Partition center channel flow extents for pre and post pylon flows
prePylonExtents, postPylonExtents = ccExtents.ySplit(pylonYPos)

# Define a uniform flow across the domain
uniformFlow = field_lib.UniformVectorField(flowVector=(-0.2, 0.5), fieldExtents=domainExtents)

# Left channel
lcVMax = 3.0
lcFlow = field_lib.DevelopedPipeFlowField(lcWidth, lcVMax, lcExtents)

# Center chanel [60, 70]
prePylonFlow = field_lib.UniformVectorField(flowVector=(0.0, 1.0), fieldExtents=prePylonExtents)
postPylonFlow = field_lib.UniformVectorField(flowVector=(0.0, 0.5), fieldExtents=postPylonExtents)
divFlow = field_lib.DivergingFlowField(1.0, (pylonXPos, 0), prePylonExtents, decay='linear')
convFlow = field_lib.ConvergingFlowField(1.0, (pylonXPos, 0), postPylonExtents, decay='linear')
ccFlow = field_lib.CompoundVectorField(prePylonFlow, postPylonFlow, divFlow, convFlow)

# Right channel [70, 100]
rcVMax = 1.5
rcFlow = field_lib.DevelopedPipeFlowField(rcWidth, rcVMax, rcExtents, offset=(70.0, 0.0))


compoundVF = field_lib.CompoundVectorField(uniformFlow, lcFlow, ccFlow, rcFlow)

with open(fileName, mode='wb') as f:
	dill.dump(compoundVF, f)