import numpy as np
import matplotlib.pyplot as plt

from ..core import vf
from ..core import utils as vf_utils
from .. import sim as vf_sim

class GridSampleComparison(object):
	""" Approximation quality evaluation via sampling both source and approximate fields
		across a grid and computing the differences between components of the vectors at
		each sample point

	"""

	def __init__(self, sampleGrid, sourceField=None, approxField=None):
		self._grid = sampleGrid
		self._source = sourceField
		self._approx = approxField

		self.invalidate()

	def invalidate(self):
		""" Invalidate/clear all computed values
		
		"""

		self._xDiff = None
		self._yDiff = None
		self._sumSquaredDiffX = None
		self._sumSquaredDiffY = None

	def changeFields(self, sourceField=None, approxField=None):
		if (sourceField is not None):
			self._source = sourceField
			self.invalidate()

		if (approxField is not None):
			self._approx = approxField
			self.invalidate()

	def changeGrid(self, sampleGrid):
		self._grid = sampleGrid
		self._invalidate()

	def plotErrors(self):
		self._fig = plt.figure(figsize=(14, 10), dpi=100)
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_title('Error Vectors')
		xGrid, yGrid = self._grid.mgrid
		magnitudes = np.sqrt(self._xDiff**2 + self._yDiff**2)
		clim = [magnitudes.min(), magnitudes.max()]
		self._q = self._ax.quiver(xGrid, yGrid, self._xDiff, self._yDiff, magnitudes, 
						clim=clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.jet)

		self._ax.axis(self._source.plotExtents)
		self._ax.minorticks_on()
		self._ax.grid(which='both', alpha=1.0, linewidth=1)
		self._ax.tick_params(which='both', direction='out')
		self._fig.colorbar(self._q, ax=self._ax)

		plt.show()
		plt.pause(10)


	def _compute(self):
		# Check if computation is possible
		if (self._source is None or self._approx is None or self._grid is None):
			self.invalidate()
			return

		xSource, ySource = self._source.sampleGrid(self._grid)
		xApprox, yApprox = self._approx.sampleGrid(self._grid)

		self._xDiff = xApprox - xSource
		self._yDiff = yApprox - ySource
		squareDiffX = self._xDiff * self._xDiff
		squareDiffY = self._yDiff * self._yDiff
		
		self._sumSquaredDiffX = np.sum(squareDiffX)
		self._sumSquaredDiffY = np.sum(squareDiffY)

	@property
	def error(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()

		return (self._sumSquaredDiffX, self._sumSquaredDiffY)

	@property
	def normalError(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()
		
		return (self._sumSquaredDiffX / self._grid.size, 
				self._sumSquaredDiffY / self._grid.size)

	@property
	def errorVectors(self):
		if (self._xDiff is None or self._yDiff is None):
			self._compute()

		return (self._xDiff, self._yDiff)



class StreamLineComparison(object):

	def __init__(self, seedParticles=None, sourceField=None, approxField=None, simTime=1, simRes=0.1):
		self._particles = seedParticles
		self._simTime = simTime
		self._simResolution = simRes

		self._source = sourceField
		self._sourceSim = vf_sim.simulators.ParticleSimulator(self._source, noise=0)

		self._approx = approxField
		self._approxSim = vf_sim.simulators.ParticleSimulator(self._approx, noise=0)
		
		self.invalidate()

	def invalidate(self):
		""" Invalidate/clear all computed values
		
		"""

		self._xDiff = None
		self._yDiff = None
		self._sumSquaredDiffX = None
		self._sumSquaredDiffY = None

	def changeFields(self, sourceField=None, approxField=None):
		if (sourceField is not None):
			self._source = sourceField
			self._sourceSim.changeField(self._source)
			self.invalidate()

		if (approxField is not None):
			self._approx = approxField
			self._approxSim.changeField(self._approx)
			self.invalidate()

	def changeParticles(self, seedParticles):
		self._particles = seedParticles
		self.invalidate()

	def _compute(self):
		# Check if computation is possible
		if (self._source is None or self._approx is None or self._particles is None):
			self.invalidate()
			return

		sourceTracks = self._sourceSim.simulate(self._particles, self._simTime, self._simResolution)
		approxTracks = self._approxSim.simulate(self._particles, self._simTime, self._simResolution)

		differences = [a - s for a, s in zip(approxTracks, sourceTracks)]
		
		self._xDiff = None
		self._yDiff = None

		for d in differences:
			diff = np.asarray(d)
			squaredDiff = diff * diff
			if (self._xDiff is None or self._yDiff is None):
				self._xDiff = diff[:, 0]
				self._yDiff = diff[:, 1]
				self._squaredDiffX = squaredDiff[:, 0]
				self._squaredDiffY = squaredDiff[:, 1]
			else:
				self._xDiff += diff[:, 0]
				self._yDiff += diff[:, 1]
			
		
		self._sumSquaredDiffX = np.sum(self._squaredDiffX)
		self._sumSquaredDiffY = np.sum(self._squaredDiffY)


	@property
	def error(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()

		return (self._sumSquaredDiffX, self._sumSquaredDiffY)

	@property
	def normalError(self):
		if (self._sumSquaredDiffX is None or self._sumSquaredDiffY is None):
			self._compute()
		
		return (self._sumSquaredDiffX / len(self._particles), 
				self._sumSquaredDiffY / len(self._particles))