import GPy

from .base import VectorFieldApproximator

class GPApproximator(VectorFieldApproximator):

	def __init__(self):
		self._measurements = []