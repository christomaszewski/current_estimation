from abc import ABCMeta, abstractmethod
from .. import vector_field as vf
from .. import core as vf_core

class VectorFieldApproximator(metaclass=ABCMeta):

	@abstractmethod
	def addMeasurement(self, measurement):
		self._measurements.append(measurement)

	@abstractmethod
	def addMeasurements(self, measurements):
		self._measurements.extend(measurements)

	@abstractmethod
	def clearMeasurements(self):
		self._measurements.clear()

	@abstractmethod
	def approximate(self):
		pass