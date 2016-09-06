import numpy as np

class Measurement(object):
	""" Represents a single measurement of a vector field at a point

	"""

	def __init__(self, point, vector):
		self.__point = point
		self.__vector = vector

	@property
	def point(self):
		return self.__point

	@property
	def vector(self):
		return self.__vector
	
	

class TrackParser(object):
	""" Converts a list of particle positions with a known timestep to 
		velocity field measurements

	"""

	def tracksToMeasurements(self, tracks, timestep):
		measurements = []
		for track in tracks:
			measurements.extend(self.trackToMeasurements(track, timestep))

		return measurements

	def trackToMeasurements(self, track, timestep):
		measurements = []
		prevPoint = None
		for point in track:
			if prevPoint is not None:
				vel = self.estimateVelocity(prevPoint, point, timestep)
				measurements.append((prevPoint, vel))

			prevPoint = point

		return measurements

	def estimateVelocity(self, p1, p2, timestep):
		velocity = ((p2[0] - p1[0]) / timestep, (p2[1] - p1[1]) / timestep) 
		return velocity		

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