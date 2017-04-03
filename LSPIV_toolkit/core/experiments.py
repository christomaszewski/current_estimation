import numpy as np
import dill
from researcher.experiment import Experiment

from .. import approx as vf_approx

class GPReconstructionExperiment(Experiment):
	""" A class representing a experiment to run in simulation

		self._field: the vector field
	"""

	def __init__(self, source, grid, evaluator):
		self._sourceVF = source
		#self._xDist = source.extents.xRange[1] - source.extents.xRange[0]
		#self._yDist = source.extents.yRange[1] - source.extents.yRange[0]

		self._xDist, self._yDist = source.extents.size

		self._evalGrid = grid
		self._evaluator = evaluator
		self._estimator = vf_approx.gp.GPApproximator()

		self._numSamples = 0
		self._numIterations = 0

	def setup(self, *args):
		self._numIterations = args[0]
		

	def run(self, *args):
		self._numSamples = args[0]
		print("starting " + str(self._numIterations) + " iterations with " + str(self._numSamples) + " samples")
		self._result = self._execute()
		return self._analyze()

	def _analyze(self):
		print("finished " + str(self._numIterations) + " iterations with " + str(self._numSamples) + " samples")
		return self._result

	def _execute(self):
		runMeanData = []
		runMinData = []
		runMaxData = []
		runApproxData = []

		for i in np.arange(self._numIterations):
			points = [tuple([np.random.rand(1)[0]*self._xDist, np.random.rand(1)[0]*self._yDist]) for _ in np.arange(self._numSamples)]
			measurements = list(self._sourceVF.measureAtPoints(points))

			self._estimator.clearMeasurements()
			self._estimator.addMeasurements(measurements)
			approxVF = self._estimator.approximate(self._sourceVF.extents)
			self._evaluator.setApprox(approxVF)
			"""(x, y) = self._evaluator.meanError
			err = np.sqrt(x**2+y**2)
			if not np.isnan(err):
				runMeanData.append(err)

			(x, y) = self._evaluator.minError
			err = np.sqrt(x**2+y**2)
			if not np.isnan(err):
				runMinData.append(err)

			(x, y) = self._evaluator.maxError
			err = np.sqrt(x**2+y**2)
			if not np.isnan(err):
				runMaxData.append(err)
			"""
			err = self._evaluator.error
			if not np.isnan(err):
				runApproxData.append(err)


		return runApproxData

