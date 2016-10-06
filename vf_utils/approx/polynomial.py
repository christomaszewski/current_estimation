from .base import VectorFieldApproximator

class PolynomialLSApproxmiator(VectorFieldApproximator):

	def __init__(self, polyDegree=2):
		""" Estimates vector field with 2nd order polynomials by default

		"""
		self._polyDegree = polyDegree
		self._measurements = []

	def addMeasurement(self, measurement):
		super().addMeasurement(measurement)

	def addMeasurements(self, measurements):
		print("Adding", str(len(measurements)), "measurements")
		super().addMeasurements(measurements)

	def clearMeasurements(self):
		super().clearMeasurements()

	def approximate(self, bounds=None):
		if (len(self._measurements) < 1):
			print("No Measurements Available")
			return None
			
		w = self.generateMonomialVector()
		monomialLength = int((self._polyDegree + 1) * (self._polyDegree + 2) / 2)

		Sx = np.zeros((monomialLength, 1))
		Sy = np.zeros((monomialLength, 1))
		S = np.zeros((monomialLength, monomialLength))
		Sxy = 0
		X = None
		vx = None
		vy = None
		for mi in self._measurements:
			pi = mi.point
			wi = w(pi)
			if(X is None):
				X = wi.transpose()
			else:
				X = np.vstack((X, wi.transpose()))

			vi = mi.vector
			if(vx is None):
				vx = vi[0]
			else:
				vx = np.vstack((vx, vi[0]))

			if(vy is None):
				vy = vi[1]
			else:
				vy = np.vstack((vy, vi[1]))
			
			Sx += vi[0]*wi
			Sy += vi[1]*wi
			S += np.dot(wi, wi.transpose())
			Sxy += vi[0]**2 + vi[1]**2

		#pseudoInvX = np.linalg.pinv(X)
		#a = np.dot(pseudoInvX, vx)
		#b = np.dot(pseudoInvX, vy)
		pseudoInvS = np.linalg.pinv(S)
		a = np.dot(pseudoInvS,Sx)
		b = np.dot(pseudoInvS,Sy)
		error = np.dot(a.transpose(),np.dot(S,a)) + np.dot(b.transpose(), np.dot(S,b))
		- 2 * np.dot(a.transpose(),Sx) - 2 * np.dot(b.transpose(),Sy) + Sxy

		print("Error: ", str(error))
		approxVF = vf.VectorField(lambda x,y: (np.dot(w((x,y)).transpose(), a)[0][0], np.dot(w((x,y)).transpose(), b)[0][0]), bounds)

		return approxVF

	def generateMonomialVector(self):
		""" Returns a lambda function that can generate the monomial vector for any inputs x and y

		"""
		degree = self._polyDegree
		vectorFunc = lambda point: np.asarray([[point[0]**xExponent * point[1]**yExponent for yExponent in range(0,degree+1) for xExponent in range(0,degree+1-yExponent)]]).transpose()

		return vectorFunc