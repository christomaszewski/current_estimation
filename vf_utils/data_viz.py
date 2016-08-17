import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

class VectorFieldView(object):
	def __init__(self, grid):
		self.__fig = plt.figure()
		self.__grid = grid
		self.__fieldArray = []


	def addField(self, vectorfield):
		self.__fieldArray.append(vectorfield)
		self.plot()

	def changeGrid(self, grid):
		self.__grid = grid
		self.plot()

	def plot(self):
		plt.clf()
		numCols = len(self.__fieldArray)
		column = 1
		for field in self.__fieldArray:
			ax = self.__fig.add_subplot(1, numCols, column)
			xGrid, yGrid = self.__grid.mgrid
			xSamples, ySamples = field.sampleAtGrid(xGrid, yGrid)
			magnitude = np.sqrt(xSamples**2 + ySamples**2)
			ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitude, cmap=cm.jet)
			ax.axis(field.bounds)
			column += 1


	def showPlots(self):
		self.plot()
		plt.show()