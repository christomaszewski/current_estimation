import numpy as np
import time

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
	

class Track(object):
	""" Represents a single particle/point on an object tracked over some 
		period of time. Can be used to produce vector field measurements

		particlePositions holds particle locations in 2-Space with the time 
		the particle was seen at the location offset from the start time of the
		track. For a point (x,y) observed at time t, the following is stored

		(t-self.__startTime, (x,y))

		startTime corresponds to time of first obervation is enforced
	"""

	def __init__(self, particlePos=None, time=None):
		self.__particlePositions = []
		self.__startTime = None

		if (particlePos is not None):
			self.__particlePositions.append((0,particlePos))
			if (time is None):
				self.__startTime = time.time()
			else:
				self.__startTime = time

	def addObservation(self, particlePos, time=None):
		if (time is None):
			time = time.time()

		if (self.__startTime is None):
			self.__startTime = time

		offsetTime = time - self.__startTime
		self.__particlePositions.append((offsetTime, particlePos))

	def getLastObservation(self):
		if (len(self.__particlePositions) < 1):
			return (None, None)

		(time, position) = self.__particlePositions[-1]
		return position

	def getMeasurements(self, method='midpoint'):
		""" Returns list of measurements representing velocity of particle
			localizing the measurement using the method specified. Velocity
			is computed by comparing pairs on consecutive points.

			midpoint: localize the measurement on the midpoint of the segment 
			between two consecutive particle locations
			front: localize measurement on first point of consecutive point pairs
			end: localize measurement on second point of consecutive point pairs

			Should return empty list of measurements if 0 or 1 observations
		"""
		methodName = "__" + method
		methodFunc = getattr(self, methodName, lambda p1, p2: p1)

		measurements = []

		prevPoint = None
		prevTime = None

		for (timestamp, point) in self.__particlePositions:
			if prevPoint is not None:
				deltaT = timestamp - prevTime
				#print(point, prevPoint)
				xVel = (point[0] - prevPoint[0]) / deltaT
				yVel = (point[1] - prevPoint[1]) / deltaT
				vel = (xVel, yVel)

				measurementPoint = methodFunc(prevPoint, point)

				m = Measurement(measurementPoint, vel)
				measurements.append(m)

			prevPoint = point
			prevTime = timestamp

		return measurements

	def __first(self, p1, p2):
		return p1

	def __last(self, p1, p2):
		return p2

	def __midpoint(self, p1, p2):
		x = (p1[0] + p2[0]) / 2
		y = (p1[1] + p2[0]) / 2
		return (x, y)
	
	def size(self):
		return len(self.__particlePositions)

	def getPointSequence(self):
		return [obs[-1] for obs in self.__particlePositions]


class TrackParser(object):
	""" Converts a list of particle positions with a known timestep to 
		velocity field measurements

		Deprecated: use track objects

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