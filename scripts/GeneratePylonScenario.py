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
pylonWidth = ccWidth
pylonYPos = yDist / 2.0
pylonStart = pylonYPos - pylonWidth / 2.0
pylonEnd = pylonYPos + pylonWidth / 2.0
pylonXPos = lcWidth + ccWidth / 2.0

pylonPartitionAxes = [pylonStart, pylonEnd]

# Partition center channel flow extents for pre and post pylon flows
prePylonExtents, pylonExtents, postPylonExtents = ccExtents.ySplit(*pylonPartitionAxes)

# Define a uniform flow across the domain
uniformFlow = field_lib.UniformVectorField(flowVector=(-0.2, 0.5), fieldExtents=domainExtents)

# Left channel
lcVMax = 3.0
lcFlow = field_lib.DevelopedPipeFlowField(lcWidth, lcVMax, lcExtents)

# Center chanel [60, 70]
prePylonFlow = field_lib.UniformVectorField(flowVector=(0.0, 1.0), fieldExtents=prePylonExtents)
postPylonFlow = field_lib.UniformVectorField(flowVector=(0.0, -0.25), fieldExtents=postPylonExtents)

# Flow around pylon
pylonDivConvExtents = vf_extents.FieldExtents((pylonXPos - pylonWidth, pylonXPos + pylonWidth),
							(0.0, yDist))
pylonDivExtents, _, pylonConvExtents = pylonDivConvExtents.ySplit(*pylonPartitionAxes)

divFlow = field_lib.DivergingFlowField(3.0, (pylonXPos, 0), pylonDivExtents, decay='linear')
convFlow = field_lib.ConvergingFlowField(3.0, (pylonXPos, 0), pylonConvExtents, decay='linear')

# Pylon flow to counteract uniform flow component
pylonFlow = field_lib.UniformVectorField(flowVector=(0.2, -0.5), fieldExtents=pylonExtents)

ccFlow = field_lib.CompoundVectorField(prePylonFlow, pylonFlow, postPylonFlow, divFlow, convFlow)

# Right channel [70, 100]
rcVMax = 1.5
rcFlow = field_lib.DevelopedPipeFlowField(rcWidth, rcVMax, rcExtents, offset=(70.0, 0.0))


compoundVF = field_lib.CompoundVectorField(uniformFlow, lcFlow, ccFlow, rcFlow)

with open(fileName, mode='wb') as f:
	dill.dump(compoundVF, f)