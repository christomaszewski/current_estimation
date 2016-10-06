import numpy as np
from matplotlib.pyplot import cm
from vf_utils.core import Measurement


class VectorField(object):
	"""Object representing a vector field

	"""

	def __init__(self, fieldFunction=None, fieldBounds=None):
		""" Object expects a lambda function that takes two variables and valid
			bounds for field
		"""
		if (fieldFunction is not None):
			self._field = fieldFunction

		self._bounds = fieldBounds

	def sampleAtPoint(self, point):
		""" Returns the value of the vector field at the 
			point provided

			todo: add bounds check
		"""

		"""if (self._bounds is not None):
			if (point[0] >= self._bounds[0] and point[0] <= self._bounds[1] and point[1] >= self._bounds[2] and point[1] <= self._bounds[3]):
				return self._field(point[0], point[1])
			else:
				return (0,0)
		"""

		return self._field(point[0], point[1])

	def sampleAtPoints(self, points):
		""" Returns the value of the vector field at each
			of the points in the list provided

			todo: check if points within bounds
		"""
		return map(self.sampleAtPoint, points)

	def sampleAtGrid(self, gridX, gridY):
		""" Returns a sampling of the vector field 
			at each point in the grid provided

			todo: check that grid lies within bounds
		"""
		wrapper = lambda x,y: self.sampleAtPoint((x,y))
		vectorizedField = np.vectorize(wrapper)

		return vectorizedField(gridX, gridY)

	def sampleGrid(self, grid):
		""" Returns a sampling of vector field at each point in grid
			
			Expects a SampleGrid object
		"""
		xGrid, yGrid = grid.mgrid
		return self.sampleAtGrid(xGrid, yGrid)

	def quiver(self, plot, grid):
		""" Produces a quiver plot of the vector field on the grid provided

			Expects a SampleGrid and a plot to call quiver on

			Deprecated - Use Data Visualization Objects
		"""
		xGrid, yGrid = grid.mgrid
		xSamples, ySamples = self.sampleAtGrid(xGrid, yGrid)
		magnitude = np.sqrt(xSamples**2 + ySamples**2)
		plot.quiver(xGrid, yGrid, xSamples, ySamples, magnitude, cmap=cm.jet)
		plot.axis(self._bounds)
		plot.grid()

	def generateMeasurementsOnGrid(self, grid):
		""" Return a list of tuples representing points and vectors at
			those points on a grid
		"""
		measurements = []
		xRange, yRange = grid.arange

		for x in xRange:
			for y in yRange:
				point = (x, y)
				vector = self.sampleAtPoint(point)
				measurements.append((point, vector))

		return measurements

	def measureAtPoint(self, point):
		return (point, self.sampleAtPoint(point))

	def setValidBounds(self, fieldBounds):
		self._bounds = fieldBounds

	@property
	def bounds(self):
		return self._bounds
	
	def __add__(self, other):
		newFunc = lambda x, y: tuple(map(sum, zip(self._field(x,y), other.sampleAtPoint((x,y)))))
		return VectorField(newFunc, self._bounds)

	def __radd__(self, other):
		newFunc = lambda x, y: tuple(map(sum, zip(self._field(x,y), other.sampleAtPoint((x,y)))))
		return VectorField(newFunc, self._bounds)

class UniformVectorField(VectorField):
	""" Standard vector field representing uniform flow in given direction
		with a given magnitude
	"""

	def __init__(self, flowVector, fieldBounds=None):
		self._field = lambda x, y: flowVector

		if (fieldBounds is not None):
			self._bounds = fieldBounds

class DevelopedPipeFlowField(VectorField):
	""" Standard vector field representing fully developed pipe flow in channel
		of specified width and specified max velocity

	"""

	def __init__(self, width, vMax, fieldBounds=None, offset=0):
		self.__channelWidth = width
		self.__maxVelocity = vMax

		self._field = lambda x, y: (0, 
			((4 * (x - offset) / self.__channelWidth - 4 * (x - offset)**2 / self.__channelWidth**2) * self.__maxVelocity))

		if (fieldBounds is not None):
			self._bounds = fieldBounds

class PieceWiseFlowField(VectorField):
	""" Represents piecewise combination of multiple fields

	"""

	def __init__(self, field1, field2, bounds1, bounds2):
		self._field1 = field1
		self._field2 = field2
		self._bounds1 = bounds1
		self._bounds2 = bounds2

	def sampleAtPoint(self, point):
		if (point[0] <= self._bounds1[1]):
			return self._field1.sampleAtPoint(point)
		else:
			return self._field2.sampleAtPoint(point)