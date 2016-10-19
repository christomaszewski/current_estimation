import numpy as np

from . import representation
from . import extents
from .base import Field
from ..utils import Measurement

class VectorField(Field):
	""" Object representing a vector field

		This is being transitioned to a base class that should not be
		instantiated! - Not true anymore?

		Todo: Need to think about implementing changing extents

	"""

	def __init__(self, fieldRepresentation):
		""" Object expects a valid vector field representation
		"""
		self._fieldRep = fieldRepresentation

	def sampleAtPoint(self, point):
		""" Returns the value of the vector field at the 
			point provided

		"""
		return self._fieldRep[point]

	def sampleAtPoints(self, points):
		""" Returns the value of the vector field at each
			of the points in the list provided

		"""
		return map(self.sampleAtPoint, points)

	def sampleAtGrid(self, gridX, gridY):
		""" Returns a sampling of the vector field 
			at each point in the grid provided

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
				measurements.append(Measurement(point, vector))

		return measurements

	def measureAtPoint(self, point):
		return Measurement(point, self.sampleAtPoint(point))


	@property
	def representation(self):
		return self._fieldRep

	@property
	def extents(self):
		return self._fieldRep.validExtents

	@property
	def plotExtents(self):
		fieldExtents = self._fieldRep.validExtents
		xRange = fieldExtents.xRange
		yRange = fieldExtents.yRange

		if (xRange is not None and yRange is not None):
			return list(xRange + yRange)

		return None
	
	def __add__(self, other):
		# todo: appropriately combine vector fields
		pass

	def __radd__(self, other):
		# todo: appropriately combine vector fields
		pass


class UniformVectorField(VectorField):
	""" Standard vector field representing uniform flow in given direction
	with a given magnitude
	"""

	def __init__(self, flowVector, fieldExtents=None):
		"""Constructs a uniform vector field over the extents given. If fieldExtents
		are None, infinite extents are used

		"""

		if (fieldExtents is None):
			fieldExtents = extents.InfiniteExtents()

		vfFunc = lambda x,y: flowVector

		self._fieldRep = representation.VectorFieldRepresentation(vfFunc, fieldExtents)

class DevelopedPipeFlowField(VectorField):
	""" Standard vector field representing fully developed pipe flow in channel
	of specified width and specified max velocity

	"""

	def __init__(self, channelWidth, vMax, fieldExtents=None, offset=(0,0)):
		"""Constructs fully developed pipe flow field along y axis, offset by offset[0],
		valid for the extents given. If extents are not provided, square extents with a
		side length equal to the channelWidth will be computed at the offset

		"""

		if (fieldExtents is None):
			xRange = (offset[0], offset[0] + channelWidth)
			yRange = (offset[1], offset[1] + channelWidth)
			fieldExtents = extents.FieldExtents(xRange, yRange)

		vfFunc = lambda x,y: (0,
			((4 * (x - offset[0]) / channelWidth - 4 * (x - offset[0])**2 / channelWidth**2) * vMax))

		self._fieldRep = representation.VectorFieldRepresentation(vfFunc, fieldExtents)

class DivergingFlowField(VectorField):
	""" A flow field diverging from a given central axis

	Note:
		Only supports vertical center axes for now

	"""

	def __init__(self, flowMag, centerAxis, fieldExtents):
		axisX, axisY = centerAxis
		
		extents = fieldExtents.xSplit(axisX)

		if (len(extents) is 2):
			vfFuncLeft = lambda x,y: (-flowMag, 0)
			vfFuncRight = lambda x,y: (flowMag, 0)

			vfLeft = representation.VectorFieldRepresentation(vfFuncLeft, extents[0])
			vfRight = representation.VectorFieldRepresentation(vfFuncRight, extents[1])
			self._fieldRep = representation.CompoundVectorFieldRepresentation(vfLeft)
			self._fieldRep.addField(vfRight)
		else:
			# Error for now, needs to create appropriate side of field in future
			print("Error, axis not in extents")


class ConvergingFlowField(VectorField):
	""" A flow field converging to a given central axis

	Note:
		Only supports vertical center axes for now

	"""

	def __init__(self, flowMag, centerAxis, fieldExtents):
		axisX, axisY = centerAxis
		
		extents = fieldExtents.xSplit(axisX)

		if (len(extents) is 2):
			vfFuncLeft = lambda x,y: (flowMag, 0)
			vfFuncRight = lambda x,y: (-flowMag, 0)

			vfLeft = representation.VectorFieldRepresentation(vfFuncLeft, extents[0])
			vfRight = representation.VectorFieldRepresentation(vfFuncRight, extents[1])
			self._fieldRep = representation.CompoundVectorFieldRepresentation(vfLeft)
			self._fieldRep.addField(vfRight)
		else:
			# Error for now, needs to create appropriate side of field in future
			print("Error, axis not in extents")


class CompoundVectorField(VectorField):
	"""Vector field object that is composed of multiple component vector fields

	"""

	def __init__(self, *args):
		"""Builds a compound vector field object from input vector fields

		"""

		if (len(args) < 1):
			# error: must give at least one field
			# todo: handle this
			return

		self._fieldRep = None
		for field in args:
			if (self._fieldRep is None):
				self._fieldRep = representation.CompoundVectorFieldRepresentation(field.representation, (0.0,0.0))
			else:
				self._fieldRep.addField(field.representation)

	def __add__(self, other):
		# todo: appropriately combine vector fields
		self._fieldRep.addField(other.representation)

	def __radd__(self, other):
		# todo: appropriately combine vector fields
		self._fieldRep.addField(other.representation)