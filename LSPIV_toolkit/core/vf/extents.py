class FieldExtents(object):
	"""Class representing the valid extents of a 2D field

	Provides methods for checking whether a given point is within the defined
	region of a field for use in computing the resultant field value at the
	point in potentially mixed and overlapping fields.

	Attributes:
		_xMin (double): Minimum x value defined
		_xMax (double): Maximum x value defined
		_yMin (double): Minimum y value defined
		_yMax (double): Maximum y value defined

	"""

	def __init__(self, xRange=(0.0, 0.0), yRange=None):
		"""Extracts and stores valid field extents

		Args:
			xRange (2-Tuple): (xMin, xMax)
			yRange (2-Tuple): (yMin, yMax) defaults to xRange when not provided
		"""

		if (yRange is None):
			yRange = xRange

		self._xMin, self._xMax = xRange
		self._yMin, self._yMax = yRange

	def contain(self, point):
		x, y = point
		if (self._xMin <= x <= self._xMax) and (self._yMin <= y <= self._yMax):
			return True

		return False

	@property
	def xRange(self):
		return (self._xMin, self._xMax)

	@property
	def yRange(self):
		return (self._yMin, self._yMax)

class PiecewiseExtents(FieldExtents):
	"""Class representing the valid extents of a 2D field defined as set of
	component subextents. Does not consider the space between component extents
	as valid.

	"""

	def __init__(self, extents=None):
		if (extents is not None):
			self._subExtents = [extents]
		else:
			self._subExtents = []

	def addExtents(self, extents):
		self._subExtents.append(extents)

	def contain(self, point):
		# same as this?
		#return any([e.contain(point) for e in self._subExtents])

		for extents in self._subExtents:
			if (extents.contain(point)):
				return True

		return False


	@property
	def xRange(self):
		return None

	@property
	def yRange(self):
		return None

class EncompassingExtents(FieldExtents):
	"""Class representing the valid extents of a 2D field which grow to
	encompass component extents as they are added. This results in the 
	space between component extents being defined as valid.

	Note:
		Does not currently handle infinite extents!!!

	"""

	def __init__(self, extents):
		self._xMin, self._xMax = extents.xRange
		self._yMin, self._yMax = extents.yRange

		self._subExtents = [extents]

	def addExtents(self, extents):
		self._subExtents.append(extents)

		xMin, xMax = extents.xRange
		yMin, yMax = extents.yRange

		self._xMin = min(self._xMin, xMin)
		self._yMin = min(self._yMin, yMin)
		self._xMax = max(self._xMax, xMax)
		self._yMax = max(self._yMax, yMax)


class InfiniteExtents(FieldExtents):
	"""Class representing infinite field extents (i.e. all points valid)

	"""

	def __init__(self):
		"""No arguments are necessary, extents are infinite

		"""
		pass

	def contain(self, point):
		return True

	@property
	def xRange(self):
		return None

	@property
	def yRange(self):
		return None