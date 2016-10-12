import numpy as np
import time

class Measurement(object):
	""" Represents a single measurement of a vector field at a point

	"""

	def __init__(self, point, vector, score=0.0):
		self._point = point
		self._vector = vector
		self._score = score

	@property
	def point(self):
		return self._point

	@property
	def vector(self):
		return self._vector
	
	@property
	def score(self):
		return self._score

	def __cmp__(self, other):
		# Negative so we can use minheap as maxheap
		return -(cmp(self.score, other.score))

	def __lt__(self, other):
		return self.score < other.score

	def __gt__(self, other):
		return self.score > other.score

	def __radd__(self, other):
		return self.score + other

	def __add__(self, other):
		return self.score + other

	def __str__(self):
		return "[" + str(self.point) + "," + str(self.vector) + "]"

	def __repr__(self):
		return str(self)

class SampleGrid(object):
	""" Convenience class to represent a grid to be sampled over.
		Also useful for plotting with quiver

		note: Dimensions all in meters or number of cells
	"""

	def __init__(self, xDistance, yDistance, xCellCount, yCellCount=None):
		self.__xDist = xDistance 							#meters
		self.__yDist = yDistance							#meters
		self.__xCellCount = xCellCount						#cells

		if (yCellCount is None):
			self.__yCellCount = xCellCount					#cells
		else:
			self.__yCellCount = yCellCount					#cells

		self.__xCellWidth = xDistance / xCellCount			#meters
		self.__xCellHalfWidth = self.__xCellWidth / 2.0		#meters
		self.__yCellWidth = yDistance / yCellCount			#meters
		self.__yCellHalfWidth = self.__yCellWidth / 2.0		#meters

		self.__xGrid, self.__yGrid = self.generateGrid()

	def generateGrid(self):
		""" Generates mgrid for used with quiver

		"""
		return np.mgrid[self.__xCellHalfWidth:(self.__xDist - self.__xCellHalfWidth):(self.__xCellCount * 1j), self.__yCellHalfWidth:(self.__yDist - self.__yCellHalfWidth):(self.__yCellCount * 1j)]


	@property
	def mgrid(self):
		return self.__xGrid, self.__yGrid

	@property
	def arange(self):
		return np.arange(self.__xCellHalfWidth, self.__xDist, self.__xCellWidth), np.arange(self.__yCellHalfWidth, self.__yDist, self.__yCellWidth)