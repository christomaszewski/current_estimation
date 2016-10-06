import numpy as np
import vf_utils.vector_field as vf
import vf_utils.core as vf_core

class ParticleSimulator(object):
	""" Todo: allow simulator to propagate existing tracks forward in time
		Todo: add errors in propagation for seed particles

	"""

	def __init__(self, vectorField):
		self.__flowField = vectorField


	def simulate(self, seedParticles, time, timestep):
		particleTracks = []
		for (timeSeen, particle) in seedParticles:
			if (timeSeen > time):
				continue
				
			track = vf_core.Track(particlePos=particle, time=timeSeen)
			for t in np.arange(timeSeen + timestep, time, timestep):
				particlePos = track.getLastObservation()
				particleVel  = self.__flowField.sampleAtPoint(particlePos)
				newParticlePos = self.propagate(particlePos, particleVel, timestep)
				track.addObservation(newParticlePos, t)

			particleTracks.append(track)

		return particleTracks

	def propagate(self, particlePos, particleVel, time):
		delta  = (particleVel[0] * time, particleVel[1] * time)
		return tuple(map(sum, zip(particlePos, delta)))