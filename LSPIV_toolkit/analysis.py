import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.ticker as ticker

class MeasurementProcessor(object):
	""" Class to handle binning measurements, discarding low score measurements
		and determining regions of low measurement availability


		todo: need valid bounds/region abstraction to be used across all code
	"""

	def __init__(self, xDistance, yDistance, xCellCount, yCellCount=None):
		self._xDist = xDistance 							#meters
		self._yDist = yDistance							#meters
		self._xCellCount = xCellCount						#cells

		if (yCellCount is None):
			self._yCellCount = xCellCount					#cells
		else:
			self._yCellCount = yCellCount					#cells

		self._xCellWidth = xDistance / xCellCount			#meters
		self._xCellHalfWidth = self._xCellWidth / 2.0		#meters
		self._yCellWidth = yDistance / self._yCellCount		#meters
		self._yCellHalfWidth = self._yCellWidth / 2.0		#meters

		self._maxMeasurementsPerCell = 2

		self._measurementBins = defaultdict(list)

		plt.ion()
		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = self._fig.add_subplot(1,1,1)
		self._fig.canvas.draw()
		self._img = None
		plt.pause(0.00001)

		self._cmap = colors.LinearSegmentedColormap.from_list('custom map', ['red','yellow','green'], 1024)

		major_ticks = np.arange(0, xCellCount+1, 5)                                              
		minor_ticks = np.arange(0, xCellCount+1, 1)                                               

		self._ax.set_xticks(major_ticks)                                                       
		self._ax.set_xticks(minor_ticks, minor=True)   

		major_ticks = np.arange(0, self._yCellCount+1, 5)                                              
		minor_ticks = np.arange(0, self._yCellCount+1, 1) 

		self._ax.set_yticks(major_ticks)                                                       
		self._ax.set_yticks(minor_ticks, minor=True)

		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * self._xCellWidth))
		self._ax.xaxis.set_major_formatter(ticks_x)

		ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y* self._yCellWidth))
		self._ax.yaxis.set_major_formatter(ticks_y)

		self._timeStamp = None

		self._annotation = ''


	def addMeasurements(self, measurements):
		# Todo accept track as input? 
		# Or since measurements come in agregated by track they are from use this 
		# for faster lookups

		#print("Filtering ", len(measurements), " measurements")

		for m in measurements:
			self.addMeasurement(m)

	def addMeasurement(self, measurement):
		coordinates = self.binMeasurement(measurement)
		#print("Adding measurment to cell ", coordinates)

		self._measurementBins[coordinates].append(measurement)
		measurementBin = self._measurementBins[coordinates]
		numMeasurements = len(measurementBin)

		if (numMeasurements > self._maxMeasurementsPerCell):
			# Trigger pruning

			worstMeasurement = None
			minScoreIndex = None

			for index in range(0, numMeasurements):
				#print(measurementBin[index].score)
				if (worstMeasurement is None or measurementBin[index] < worstMeasurement):
					minScoreIndex = index
					worstMeasurement = measurementBin[index]

			del self._measurementBins[coordinates][minScoreIndex]

	def getMeasurements(self):
		measurements = []

		for key in self._measurementBins:
			measurements.extend(self._measurementBins[key])

		return measurements

	def clearMeasurements(self):
		""" Should rarely be used but implemented for now to simplify running simulations
			where intermediate results are desired

		"""
		self._measurementBins.clear()

	def binMeasurement(self, measurement):
		""" Determine which cell the measurement should go in

		"""
		(x, y) = measurement.point

		xCell = int(np.floor(x / self._xCellWidth))
		yCell = int(np.floor(y / self._yCellWidth))

		return (xCell, yCell)

	def drawMeasurementGrid(self):
		# Need this to actually get the plots to update
		plt.pause(0.0001)

		grid = np.zeros((self._yCellCount, self._xCellCount), dtype=np.float64)

		for (x,y) in self._measurementBins:
			if (0 > x or x >= self._xCellCount):
				print("x index of measurement out of bounds: ", (x,y))
				continue
			if (0 > y or y >= self._yCellCount):
				print("y index of measurement out of bounds: ", (x,y))
				continue
			if (self._measurementBins[(x,y)] is not None):
				# use number of measurements in cell
				#grid[y, x] = len(self._measurementBins[(x,y)])

				# use avg of scores in cell
				grid[y, x] = sum(self._measurementBins[(x,y)]) / len(self._measurementBins[(x,y)])
				#print(grid[y,x])

		#print(grid)
		self._ax.grid(which='both', alpha=1.0, color='white', linewidth=1)

		
		if (self._img is None):
			self._img = self._ax.imshow(grid,cmap=self._cmap,interpolation='nearest', origin='lower', extent=(0, self._xCellCount, 0, self._yCellCount), aspect='auto')
			self._cbar = plt.colorbar(self._img, ax=self._ax, cmap=self._cmap)	
			self._cbar.set_clim(0, self._maxMeasurementsPerCell)
			self._cbar.draw_all()
		else:
			self._img.set_data(grid)


		self._fig.canvas.draw()
		#plt.draw()

	def setAnnotation(self, text):
		self._annotation = text

	def saveFig(self, filename, timestamp=None):

		if (self._timeStamp is not None):
			self._timeStamp.remove()

		#self._timeStamp = self._ax.text(17, 19, annotation)

		self.drawMeasurementGrid()
		self._ax.set_title('Measurement Distribution' + self._annotation)
		self._ax.grid(which='both', alpha=1.0, color='white', linewidth=1)
		self._ax.tick_params(which='both', direction='out')
		self._fig.savefig(filename, bbox_inches='tight', dpi=100)