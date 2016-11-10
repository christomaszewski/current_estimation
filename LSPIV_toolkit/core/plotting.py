import matplotlib.pyplot as plt
import numpy as np
import cv2

plt.ion()

class SimpleFieldView(object):
	"""Basic vector field plotting functionality

	"""

	def __init__(self, field=None, grid=None, pause=0.0001):
		self._field = field
		self._grid = grid

		plt.ion()
		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = None
		self._q = None

		self._pauseLength = pause

		self._title = "Untitled"
		self._clim = None
		self._annotation = ""

	def draw(self):
		plt.show()
		plt.pause(self._pauseLength)

	def setTitle(self, title):
		self._title = title

	def setClim(self, lim):
		self._clim = lim

	def setAnnotation(self, text):
		self._annotation = text

	def quiver(self):
		if (self._field is None or self._grid is None):
			return

		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title(self._title + self._annotation)
		#self._ax.grid(True)

		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = self._field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)

		if (self._clim is None):
			self._clim = [magnitudes.min(), magnitudes.max()]

		self._q = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)


		#self._ax.text(85, 48, self._annotation)

		self._ax.axis(self._field.plotExtents)

		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')

		self._fig.colorbar(self._q, ax=self._ax)


		self.draw()

		# Force recomputation of colorbar
		self._clim = None

	def plotTrack(self, track, color, marker='o'):
		t = np.asarray(track.getPointSequence())
		self._ax.scatter(t[:,0], t[:,1], c=color, marker=marker)
		self.draw()

	def changeGrid(self, newGrid):
		self._grid = newGrid
		self.quiver()

	def changeField(self, newField):
		self._field = newField
		self.quiver()

	def save(self, fileName="default.png"):
		self._fig.savefig(fileName, bbox_inches='tight', dpi=100)

	@property
	def clim(self):
		return self._clim


class OverlayFieldView(object):
	"""Plotting a vector field over a background image

	"""

	def __init__(self, field=None, grid=None, img=None, pause=0.0001):
		self._field = field
		self._grid = grid

		# Assume incoming image is in opencv format
		if (img is not None):
			self.updateImage(img)

		plt.ion()
		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = None
		self._q = None

		self._pauseLength = pause

		self._title = "Untitled"
		self._clim = None
		self._annotation = ""

	def updateImage(self, img):
		self._img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self._img = np.flipud(self._img)

	def draw(self):
		plt.show()
		plt.pause(self._pauseLength)

	def setTitle(self, title):
		self._title = title

	def setClim(self, lim):
		self._clim = lim

	def setAnnotation(self, text):
		self._annotation = text

	def quiver(self):
		if (self._field is None or self._grid is None):
			return

		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title(self._title + self._annotation)

		if (self._img is not None):
			self._ax.imshow(self._img, origin='lower', aspect='auto')

		self._ax.hold(True)

		#self._ax.grid(True)

		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = self._field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)

		if (self._clim is None):
			self._clim = [magnitudes.min(), magnitudes.max()]

		self._q = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)


		#self._ax.text(85, 48, self._annotation)

		self._ax.axis(self._field.plotExtents)

		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')

		self._fig.colorbar(self._q, ax=self._ax)

		self.draw()

		# Force recomputation of colorbar
		self._clim = None

	def plotTrack(self, track, color, marker='o'):
		t = np.asarray(track.getPointSequence())
		self._ax.scatter(t[:,0], t[:,1], c=color, marker=marker)
		self.draw()

	def changeGrid(self, newGrid):
		self._grid = newGrid
		self.quiver()

	def changeField(self, newField):
		self._field = newField
		self.quiver()

	def save(self, fileName="default.png"):
		self._fig.savefig(fileName, bbox_inches='tight', dpi=100)

	@property
	def clim(self):
		return self._clim