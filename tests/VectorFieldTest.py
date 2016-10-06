import numpy as np
import matplotlib.pyplot as plt
import vf_utils.vector_field as vf
import vf_utils.approximate as vf_approx
import vf_utils.core as vf_core
import vf_utils.data_viz as vf_viz

vMax = 3 #m/s
riverWidth = 100 #meters

vMax1 = 3 #m/s
riverWidth1 = 40 #meters

vMax2 = 4 #m/s
riverWidth2 = 60 #meters

vfBounds1 = [0, riverWidth1, 0, riverWidth1 + riverWidth2]

vfBounds2 = [riverWidth1, riverWidth2, 0, riverWidth1 + riverWidth2]

vfBounds = [0, riverWidth1 + riverWidth2, 0, riverWidth1 + riverWidth2]

sourceVF1 = vf.DevelopedPipeFlowField(riverWidth1, vMax1, vfBounds1, 0)# + vf.UniformVectorField((0.2,0), vfBounds)

sourceVF2 = vf.DevelopedPipeFlowField(riverWidth2, vMax2, vfBounds2, riverWidth1)# + vf.UniformVectorField((-0.3,0), vfBounds)


sourceVF = vf.DevelopedPipeFlowField(riverWidth, vMax, vfBounds, 0)
#vf.PieceWiseFlowField(sourceVF1, sourceVF2, vfBounds1, vfBounds2)
#sourceVF.setValidBounds(vfBounds)

xGrid = 25 #cells
yGrid = 8 #cells
xDist = 100 #meters
yDist = 100 #meters

sampleGrid = vf_core.SampleGrid(xDist, yDist, xGrid, yGrid)

fieldView = vf_viz.VectorFieldView(sampleGrid)

fieldView.addField(sourceVF)
fieldView.plot()
#fieldView.showPlots()
fieldView.saveFig("scenario1.png")