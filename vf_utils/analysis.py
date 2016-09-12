import numpy as np

class MeasurementProcessor(object):
	""" Class to handle binning measurements, discarding low score measurements
		and determining regions of low measurement availability


		todo: need valid bounds/region abstraction to be used across all code
	"""

	def __init__(object, bounds):
		self.__bounds = bounds