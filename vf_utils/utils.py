import numpy as np

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