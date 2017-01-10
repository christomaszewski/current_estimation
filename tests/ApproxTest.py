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

if __name__ == '__main__':
	plt.ion()

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
	displayGrid = vf_utils.SampleGrid(xDist, yDist, xGrid+15, yGrid+5)

	sourceFieldView = vf_plot.SimpleFieldView(compoundVF, sampleGrid)
	sourceFieldView.setTitle('Sampled Source Field')

	sourceFieldView.quiver()

	approxFieldView = vf_plot.SimpleFieldView(grid=sampleGrid)
	approxFieldView.setTitle('GP Approximation')
	approxFieldView.setClim(sourceFieldView.clim)

	seedParticles = [(0, p) for p in displayGrid.cellCenters]

	streamEval = vf_approx.eval.StreamLineComparison(seedParticles=seedParticles,
					sourceField=compoundVF, simTime=5, simRes=0.1)
	sampleEvaluator = vf_approx.eval.GridSampleComparison(sampleGrid, sourceField=compoundVF)
	displayEvaluator = vf_approx.eval.GridSampleComparison(displayGrid, sourceField=compoundVF)

	vfEstimator = vf_approx.gp.GPApproximator()

	measurements = compoundVF.generateMeasurementsOnGrid(sampleGrid)
	vfEstimator.addMeasurements(measurements)

	approxVF = vfEstimator.approximate(compoundVF.extents)

	sampleEvaluator.changeFields(sourceField=compoundVF, approxField=approxVF)
	displayEvaluator.changeFields(sourceField=compoundVF, approxField=approxVF)
	streamEval.changeFields(sourceField=compoundVF, approxField=approxVF)

	print("Sample Sum Squared Error: ", sampleEvaluator.error)
	print("Sample Normal Error: ", sampleEvaluator.normalError)
	print("Sample Min Error: ", sampleEvaluator.minError)
	print("Sample Max Error: ", sampleEvaluator.maxError)
	print("Sample Mean Error: ", sampleEvaluator.meanError)
	print("Sample Standard Dev of Error: ", sampleEvaluator.errorStd)

	print("Display Error: ", displayEvaluator.normalError)
	print("Display Sum Squared Error: ", displayEvaluator.error)
	print("Display Normal Error: ", displayEvaluator.normalError)
	print("Display Min Error: ", displayEvaluator.minError)
	print("Display Max Error: ", displayEvaluator.maxError)
	print("Display Mean Error: ", displayEvaluator.meanError)
	print("Display Standard Dev of Error: ", displayEvaluator.errorStd)

	print("Normal Stream Error: ", streamEval.normalError)
	print("Stream Error: ", streamEval.error)

	streamEval.plotErrors(displayGrid)
	streamEval.save('../output/stream_sample_approx.png')

	streamEval.plotErrors(displayGrid, field='source')
	streamEval.save('../output/stream_sample_source.png')

	streamEval.plotStreamlineComparison()
	streamEval.save('../output/streamline_comp.png')


	sampleEvaluator.plotErrors()
	sampleEvaluator.save('../output/sample_errors.png')
	displayEvaluator.plotErrors()
	displayEvaluator.save('../output/display_errors.png')
	
	approxFieldView.changeField(approxVF)
	approxFieldView.quiver()

	sourceFieldView.save('../output/source.png')
	approxFieldView.save('../output/approx.png')

	sourceFieldView.changeGrid(displayGrid)
	approxFieldView.changeGrid(displayGrid)

	sourceFieldView.setTitle('Densely Sampled Source Field')
	approxFieldView.setTitle('Dense GP Approximation')

	sourceFieldView.quiver()
	approxFieldView.quiver()

	sourceFieldView.save('../output/source_dense.png')
	approxFieldView.save('../output/approx_dense.png')

	with open('../scenarios/approx_pylon.scenario', mode='wb') as f:
		dill.dump(approxVF, f)

	plt.pause(5)