import matplotlib.pyplot as plt
import numpy as np
import dill

from context import LSPIV_toolkit

import LSPIV_toolkit.core.vf.fields as field_lib
import LSPIV_toolkit.core.vf.extents as vf_extents

""" Script to generate a flow simulating a fully developed single channel flow
"""

# Scenario Output filename
fileName = "../scenarios/single_channel.scenario"

channelWidth = 100
vMax = 3.0

# Display Grid Definition
xGrid = 50 #cells
yGrid = 10 #cells
xDist = channelWidth #meters
yDist = 50 #meters

# Domain Extents [0, 100, 0, 50]
domainExtents = vf_extents.FieldExtents((0.0, xDist), (0.0, yDist))

channelFlow = field_lib.DevelopedPipeFlowField(channelWidth, vMax, domainExtents)

with open(fileName, mode='wb') as f:
	dill.dump(channelFlow, f)