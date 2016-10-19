import GPy
import numpy as np

from ..core import vf
from .base import VectorFieldApproximator

class GPApproximator(VectorFieldApproximator):

	def __init__(self, kernel=None):
		self._measurements = []

		if (kernel is None):
			# Default kernel
			self._K = GPy.kern.Matern52(input_dim=2, ARD=True, lengthscale=10)
		else:
			self._K = kernel

		self._gpModelX = None
		self._gpModelY = None


	def addMeasurement(self, measurement):
		super().addMeasurement(measurement)

	def addMeasurements(self, measurements):
		super().addMeasurements(measurements)

	def clearMeasurements(self):
		super().clearMeasurements()

	def approximate(self, fieldExtents=None):
		if (len(self._measurements) < 1):
			print("No Measurements Available")
			return None

		X = []
		vX = []
		vY = []

		for m in self._measurements:
			X.append(m.point)
			vel = m.vector
			vX.append((vel[0]))
			vY.append((vel[1]))


		x = np.asarray(X)
		y1 = np.asarray(vX)
		y2 = np.asarray(vY)
		y1 = np.reshape(y1, (len(vX),1))
		y2 = np.reshape(y2, (len(vY),1))


		self._gpModelX = GPy.models.GPRegression(x, y1, self._K, normalizer=False)
		self._gpModelY = GPy.models.GPRegression(x, y2, self._K, normalizer=False)

		self._gpModelX.optimize(max_f_eval = 1000)
		self._gpModelY.optimize(max_f_eval = 1000)

		#self._gpModel.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})

		vfRep = vf.gp_representation.GPVectorFieldRepresentation(self._gpModelX, self._gpModelY, fieldExtents)

		return vf.fields.VectorField(vfRep)


class CoregionalizedGPApproximator(VectorFieldApproximator):

	def __init__(self):
		self._measurements = []

		self._K = GPy.kern.Matern32(input_dim=2, ARD=True, lengthscale=1)
		self._coregionalizedK = GPy.util.multioutput.ICM(input_dim=2, num_outputs=2, kernel=self._K)

		self._gpModel = None

	def addMeasurement(self, measurement):
		super().addMeasurement(measurement)

	def addMeasurements(self, measurements):
		super().addMeasurements(measurements)

	def clearMeasurements(self):
		super().clearMeasurements()

	def approximate(self, fieldExtents=None):
		if (len(self._measurements) < 1):
			print("No Measurements Available")
			return None

		X = []
		vX = []
		vY = []

		for m in self._measurements:
			X.append(m.point)
			vel = m.vector
			vX.append((vel[0]))
			vY.append((vel[1]))

		x = np.asarray(X)
		y1 = np.asarray(vX)
		y2 = np.asarray(vY)
		y1 = np.reshape(y1, (len(vX),1))
		y2 = np.reshape(y2, (len(vY),1))

	
		print(y1)
		print(y2)

		# Coregionalization stuff
		self._gpModel = GPy.models.GPCoregionalizedRegression([x, x], [y1, y2], self._coregionalizedK)
		self._gpModel['.*Mat32.var'].constrain_fixed(1.)
		self._gpModel.optimize(messages=False)

		vfRep = vf.gp_representation.CoregionalizedGPFieldRepresentation(self._gpModel, fieldExtents)

		return vf.fields.VectorField(vfRep)