import numpy as np

from .. import core as vf_core

class ParticleSimulator(object):
	""" Todo: allow simulator to propagate existing tracks forward in time
		Todo: add errors in propagation for seed particles

	"""

	def __init__(self, vectorField=None, noise=0.0001):
		self._flowField = vectorField

		# todo make this gaussian noise
		self._observationNoise = noise


	def simulate(self, seedParticles, time, timestep):
		if (self._flowField is None):
			return []

		particleTracks = []
		for (timeSeen, particle) in seedParticles:
			if (timeSeen >= time):
				continue
			
			track = vf_core.tracking.Track(position=particle, time=timeSeen)
			for t in np.arange(timeSeen + timestep, time, timestep):
				_, particlePos = track.getLastObservation()
				particleVel  = self._flowField.sampleAtPoint(particlePos)
				newParticlePos = self.propagate(particlePos, particleVel, timestep)
				
				randX = 0.5 - np.random.random()
				randY = 0.5 - np.random.random()

				observedParticlePos = tuple(map(sum, zip(newParticlePos, 
					(randX*self._observationNoise, randY*self._observationNoise))))

				track.addObservation(newParticlePos, t)

			particleTracks.append(track)

		return particleTracks

	def propagate(self, particlePos, particleVel, time):
		delta  = (particleVel[0] * time, particleVel[1] * time)
		return tuple(map(sum, zip(particlePos, delta)))

	def changeField(self, newField):
		self._flowField = newField