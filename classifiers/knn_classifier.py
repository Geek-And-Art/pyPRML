import numpy as np

class KNearestNeighbor(object):
	"""
	K-Nearest Neighbor classifier will choose input data
	point's nearest k points with predefined distance 
	measure first. 

	Then find these k points' corresponding class. 

	Finally, use the class with largest counting number 
	in the k nearest points as its prediction.
	"""

	def __init__(self):
		"""
		Lazy evaluation.
		"""
		self.W = {}

	def train(self, X, Y):
		"""
		For knn, the training process is just
		remembering the training data set itself.

		Inputs:
		- X: A numpy array observed data of shape (N, D)
		- Y: Target numpy array of shape (N, 1)
		"""
		self.W['X_train'] = X
		self.W['Y_train'] = Y

	def compute_Euclidean_distance(self, X):
		"""
		Assistant method for prediction process. Use 
		Euclidian measure to compute distance.

		Inputs:
		- X: A numpy array observed data of shape (N, D). 

		     Here we need to pay attension: the 'N' here is 
		     different from 'self.W['X_train'].shape[0]'. 
		     To be clear, X's shape is (num_test, D), and
		     self.W['X_train']'s shape is (num_train, D).


		Output:
		- dists: A numpy array of shape (num_test, num_train).
		        Each point (i, j) represent the distance between
		        the ith testing data point and jth training data
		        point.
		"""

		# As python is more efficient for matrix operation like
		# matlab, we should try to vectorize our computations
		# as much as possible.
		# 
		# Here we use the fact that '(a - b)^2 = a^2 - 2ab + b^2'
		# to vectorize our computation.
		X_train = self.W['X_train']
		XTrainSquare = np.sum(X_train ** 2, axis=1)
		XTestSquare = np.sum(X ** 2, axis=1)		
		crossTerm = 2 * X.dot(X_train.T)


		# The broadcast will apply following rule: 'row matches row',
		# and 'column matches column'.
		# 
		# Thus when we use 'XTrainSquare - crossTerm', it's fine
		# because the row length of 'XtrainSquare' is equal to
		# row length of 'crossTerm'.
		# 
		# But we'll fail when we use 'XTestSquare - crossTerm',
		# because of the mismatch of row length of 'XTestSquare'
		# and 'crossTerm'.
		# 
		# On the other hand, we can make 
		# 'XTestSquare[:, np.newaxis] - crossTerm' 
		# success due to the match of column length.
		dists = np.sqrt(XTrainSquare - crossTerm + XTestSquare[:, np.newaxis])

		return dists

	def predict(self, X, k=1):
		"""
		Find the X's k nearest points by using Euclidian
		distance. And use the largest counting number 
		class as prediction

		Inputs:
		- X: A numpy array observed data of shape (N, D)
		- k: Hpyter parameter of knn used to determine how
		     many nearest points should be chosen.

		Return a vector:
		- Y: The prediction of each training data.
		"""
		dists = self.compute_Euclidean_distance(X)

		# Use 'np.argsort' to find k nearest points
		num_test = X.shape[0]
		Y_pred = np.zeros(num_test)
		Y_train = self.W['Y_train']

		for i in xrange(num_test):
			kNearestIndx = np.argsort(dists[i])[: k]
			kNearestY = Y_train[kNearestIndx]

			val, freq = np.unique(kNearestY, return_counts=True)
			Y_pred[i] = val[np.argmax(freq)]

		return Y_pred

	def loss(self):
		"""
		As there's no update process for knn,
		the loss function should be empty.
		"""
		pass
