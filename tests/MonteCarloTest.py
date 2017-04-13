import matplotlib.pyplot as plt
import numpy as np
import dill

from context import LSPIV_toolkit

from field_toolkit.core import primitives
from field_toolkit.viz.plotting import SimpleFieldView
from field_toolkit.analysis.eval import GridSampleEvaluator
import LSPIV_toolkit.core.experiments as vf_experiments

from researcher.executor import MultiProcessExecutor

if __name__ == '__main__':
	plt.ion()

	# Load Scenario
	scenarioName = 'pylon'

	# Scenario Field file name
	scenarioFile = '../scenarios/' + scenarioName + '.scenario'

	with open(scenarioFile, mode='rb') as f:
		sourceVF = dill.load(f)

	# All distances in meters
	xDist, yDist = sourceVF.extents.size

	# Define grid to sample sub m^2 cells
	xGrid = xDist*4 #cells
	yGrid = yDist*4 #cells

	grid = primitives.SampleGrid(xDist, yDist, xGrid, yGrid)

	# Initial Plot
	#sourceFieldView = SimpleFieldView(compoundVF, grid)
	#sourceFieldView.quiver()

	nIterations = 5

	evaluator = GridSampleEvaluator(grid, sourceField=sourceVF)

	exp = vf_experiments.GPReconstructionExperiment(sourceVF, grid, evaluator)
	exp.setup(nIterations)

	meanData = []
	minData = []
	maxData = []
	approxData = []

	inputArgs = np.arange(5, 65, 5)

	multiExec = MultiProcessExecutor(exp, numProcesses=4)
	multiExec.setup(inputArgs)

	result = multiExec.start()

	while(not result.ready()):
		print("Waiting for results...")
		result.wait(timeout=10)


	with open('approxData1K.dat', mode='wb') as f:
		dill.dump(result.get(), f)

	with open('sampleSize1K.dat', mode='wb') as f:
		dill.dump(inputArgs, f)