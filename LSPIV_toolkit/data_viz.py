import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cm
import numpy as np

class VectorFieldView(object):
	""" Deprecated, used core.plotting module

	"""
	def __init__(self, grid):
		self.__fig = plt.figure()
		self.__grid = grid
		self.__fieldArray = []
		self.__lastAx = None
		self._timeStamp = None
		
	def addField(self, vectorfield):
		self.__fieldArray.append(vectorfield)
		self.plot()

	def clearFields(self):
		self.__fieldArray.clear()
		#self.__fig = plt.figure()
		#self.plot()

	def changeGrid(self, grid):
		self.__grid = grid
		self.plot()

	def plot(self):
		#self.__fig.show()
		self.__fig.clf()
		#plt.clf()
		numCols = len(self.__fieldArray)
		column = 1
		for field in self.__fieldArray:
			ax = self.__fig.add_subplot(1, numCols, column)
			xGrid, yGrid = self.__grid.mgrid
			xSamples, ySamples = field.sampleAtGrid(xGrid, yGrid)
			magnitude = np.sqrt(xSamples**2 + ySamples**2)
			ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitude, cmap=cm.jet)
			ax.axis(field.bounds)
			self.__lastAx = ax
			column += 1
		#self.__fig.canvas.draw_idle()
		#self.__fig.canvas.flush_events()
		#plt.show()
		#plt.draw()

	def plotTrack(self, track, color):
		self.__lastAx.hold(True)
		t = np.asarray(track.getPointSequence())
		self.__lastAx.scatter(t[:,0], t[:,1], c=color)
		#self.plot()

	def plotBoatTrack(self, track):
		self.__lastAx.hold(True)
		t = np.asarray(track.getPointSequence())
		self.__lastAx.scatter(t[:,0], t[:,1], c=color)


	def showPlots(self):
		#self.plot()
		plt.show()

	def closePlots(self):
		plt.close(self.__fig)
		self.__fig = plt.figure()

	def saveFig(self, filename, timestamp=None):
		annotation = "Time: "
		if (timestamp is not None):
			annotation += str(timestamp)
		else:
			annotation = ""
		
		self.__lastAx.text(85, 95, annotation)
		self.__fig.savefig(filename, bbox_inches='tight')