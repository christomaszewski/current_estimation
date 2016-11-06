import matplotlib.pyplot as plt
import numpy as np
import dill

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.vf.extents as vf_extents

""" Script to generate a flow simulating two channel flows side by side
"""

# Scenario Output filename
fileName = "../scenarios/twin_channel.scenario"


# Display Grid Definition
xGrid = 50 #cells
yGrid = 10 #cells
xDist = 100 #meters
yDist = 50 #meters

# Domain Extents [0, 100, 0, 50]
domainExtents = vf_extents.FieldExtents((0.0, xDist), (0.0, yDist))

# Define channels making up the flow
lcWidth = 45.0
rcWidth = xDist - lcWidth

# Compute partition axes (these should be sorted)
partitionAxes = [lcWidth]

# Parition domain extents into channel extents
lcExtents, rcExtents = domainExtents.xSplit(*partitionAxes)

# Define a uniform flow across the domain
uniformFlow = field_lib.UniformVectorField(flowVector=(0.1, 0.5), fieldExtents=domainExtents)

# Left channel
lcVMax = 2.0
lcFlow = field_lib.DevelopedPipeFlowField(lcWidth, lcVMax, lcExtents)

# Right channel [70, 100]
rcVMax = 1.0
rcFlow = field_lib.DevelopedPipeFlowField(rcWidth, rcVMax, rcExtents, offset=(lcWidth, 0.0))

compoundVF = field_lib.CompoundVectorField(uniformFlow, lcFlow, rcFlow)

with open(fileName, mode='wb') as f:
	dill.dump(compoundVF, f)