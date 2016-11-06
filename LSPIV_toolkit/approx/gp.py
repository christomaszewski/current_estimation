import GPy
import numpy as np

from ..core import vf
from .base import VectorFieldApproximator

class GPApproximator(VectorFieldApproximator):

	def __init__(self, kernel=None):
		self._measurements = []

		if (kernel is None):
			# Default kernel
			self._Kx = GPy.kern.Matern32(2, ARD=True, lengthscale=1)
			self._Ky = (GPy.kern.Matern32(2, ARD=True, lengthscale=3) +\
						GPy.kern.Matern32(2, ARD=True, lengthscale=10)) *\
						GPy.kern.Bias(input_dim=2)
		else:
			self._Kx = kernel
			self._Ky = kernel

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

		print("Processing ", len(self._measurements), " Measurements")


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


		self._gpModelX = GPy.models.GPRegression(x, y1, self._Kx, normalizer=None)
		self._gpModelY = GPy.models.GPRegression(x, y2, self._Ky, normalizer=None)

		#print(self._gpModelX)
		#print(self._gpModelY)
		self._gpModelX.randomize()
		self._gpModelY.randomize()

		self._gpModelX.optimize_restarts(messages=False, optimizer='scg', robust=True, num_restarts=2, max_iters=100000)
		self._gpModelY.optimize_restarts(messages=False, optimizer='scg', robust=True, num_restarts=2, max_iters=100000)
		#self._gpModelX.optimize_SGD()
		#self._gpModelY.optimize_SGD()
		#print(self._gpModelX)
		#print(self._gpModelY)

		#self._gpModel.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})

		vfRep = vf.gp_representation.GPVectorFieldRepresentation(self._gpModelX, self._gpModelY, fieldExtents)

		return vf.fields.VectorField(vfRep)


class CoregionalizedGPApproximator(VectorFieldApproximator):

	def __init__(self):
		self._measurements = []


		# Bias Kernel
		self._biasK = GPy.kern.Bias(input_dim=2)
		
		# Linear Kernel
		self._linearK = GPy.kern.Linear(input_dim=2, ARD=True)
		
		# Matern 3/2 Kernel
		self._maternK = GPy.kern.Matern32(input_dim=2, ARD=True, lengthscale=5)
		
		kList = [self._biasK, self._maternK]

		# Build Coregionalized
		self._coregionalizedK = GPy.util.multioutput.LCM(input_dim=2, num_outputs=2, kernels_list=kList)

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

		print("Processing ", len(self._measurements), " Measurements")

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

		# Coregionalization stuff
		self._gpModel = GPy.models.GPCoregionalizedRegression([x, x], [y1, y2], self._coregionalizedK)
		#print(self._gpModel)

		self._gpModel.randomize()

		# Set constraints
		self._gpModel['.*ICM.*var'].unconstrain()
		self._gpModel['.*ICM0.*var'].constrain_fixed(1.)
		self._gpModel['.*ICM0.*W'].constrain_fixed(0)
		self._gpModel['.*ICM1.*var'].constrain_fixed(1.)
		self._gpModel['.*ICM1.*W'].constrain_fixed(0)
		#self._gpModel['.*ICM2.*var'].constrain_fixed(1.)


		#print(self._gpModel)

		#self._gpModel.optimize(messages=False, max_iters=10000, optimizer='lbfgsb')
		self._gpModel.optimize_restarts(num_restarts=1, max_iters=100000, optimizer='scg')
		#print(self._gpModel)
		vfRep = vf.gp_representation.CoregionalizedGPFieldRepresentation(self._gpModel, fieldExtents)

		return vf.fields.VectorField(vfRep)