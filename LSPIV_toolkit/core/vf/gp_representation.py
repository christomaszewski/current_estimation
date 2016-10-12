import GPy
import numpy as np

from . import extents
from .base import FieldRepresentation

class GPVectorFieldRepresentation(FieldRepresentation):

	def __init__(self, xModel, yModel, fieldExtents, undefinedValue=(0,0)):
		self._xComponentGPModel = xModel
		self._yComponentGPModel = yModel
		self._validExtents = fieldExtents
		self._undefinedVal = undefinedValue

	def __getitem__(self, index):
		testPoint = np.asarray([index])

		muX, varX = self._xComponentGPModel.predict_noiseless(testPoint)
		muY, varY = self._yComponentGPModel.predict_noiseless(testPoint)

		return (muX[0][0], muY[0][0])

	def isDefinedAt(self, point):
		return self._validExtents.contain(point)

	@property
	def validExtents(self):
		return self._validExtents
