import matplotlib.pyplot as plt
import numpy as np

plt.ion()

class SimpleFieldView(object):
	"""Basic vector field plotting functionality

	"""

	def __init__(self, field=None, grid=None, pause=0.0001):
		self._field = field
		self._grid = grid

		plt.ion()
		self._fig = plt.figure()
		self._ax = None
		self._q = None

		self._pauseLength = pause

	def draw(self):
		plt.show()
		plt.pause(self._pauseLength)

	def quiver(self):
		if (self._field is None or self._grid is None):
			return

		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)

		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = self._field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)
		
		self._q = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)

		self._ax.axis(self._field.plotExtents)

		self._fig.colorbar(self._q, ax=self._ax)

		self.draw()

	def changeGrid(self, newGrid):
		self._grid = newGrid
		self.quiver()

	def changeField(self, newField):
		self._field = newField
		self.quiver()

	def save(self, fileName="default.png"):
		self._fig.savefig(fileName, bbox_inches='tight')