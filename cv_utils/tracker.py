from abc import ABCMeta, abstractmethod

import numpy as np
import cv2

import vf_utils.core as vf_core

class Tracker(metaclass=ABCMeta):

	@abstractmethod
	def processImage(self, img, timestamp):
		pass

	@abstractmethod
	def getTracks(self):
		pass


class LKOpticalFlowTracker(Tracker):

	def __init__(self, lkParams, featureParams, detectionInterval=0.1):
		self.__lkParams = lkParams
		self.__featureParams = featureParams

		self.__prevImg = None
		self.__prevTimestamp = None

		self.__activeTracks = []
		self.__detectionInterval = detectionInterval
		self.__prevDetectionTime = None

		self.__deviationThreshold = 1

	def processImage(self, img, timestamp):
		# Todo: Check if input is already grayscale
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		trackEndpoints = self.getTrackEndpoints()

		newTracks = []

		# If features have never been detected or detectionInverval has lapsed
		if (self.__prevDetectionTime is None or timestamp - self.__prevDetectionTime > self.__detectionInterval):
			print("Finding New Features")
			self.__prevDetectionTime = timestamp
			searchMask = np.zeros_like(grayImg)
			searchMask[:] = 255

			# Mask all the current track end points
			for (x,y) in trackEndpoints:
				cv2.circle(searchMask, (x,y), 5, 0, -1)

			p = cv2.goodFeaturesToTrack(grayImg, mask=searchMask, **self.__featureParams)

			if (p is not None):
				for x, y in np.float32(p).reshape(-1, 2):
					newTracks.append(vf_core.Track((x,y), timestamp))

		if (self.__prevImg is not None):
			prevPoints = np.float32(trackEndpoints).reshape(-1,1,2)

			# Run LK forwards
			nextPoints, status, error = cv2.calcOpticalFlowPyrLK(self.__prevImg,
				grayImg, prevPoints, None, **self.__lkParams)

			# Run LK in reverse
			prevPointsRev, status, error = cv2.calcOpticalFlowPyrLK(grayImg,
				self.__prevImg, nextPoints, None, **self.__lkParams)

			# Compute deviation between original points and back propagation
			dev = abs(prevPoints-prevPointsRev).reshape(-1,2)

			# For each pair of points, take max deviation in either axis
			maxDev = dev.max(-1)

			# Check against max deviation threshold allowed
			matchQuality = maxDev < self.__deviationThreshold

			for (track, point, match) in zip(self.__activeTracks, nextPoints.reshape(-1,2), matchQuality):
				if (match):
					track.addObservation(point, timestamp)
					newTracks.append(track)

				# Todo: Handle tracks that have been lost


		self.__activeTracks = newTracks

		self.__prevImg = grayImg
		self.__prevTimestamp = timestamp


	def getTrackEndpoints(self):
		endpoints = []

		for track in self.__activeTracks:
			position = track.getLastObservation()
			endpoints.append(position)

		return endpoints

	def getTracks(self):
		return self.__activeTracks