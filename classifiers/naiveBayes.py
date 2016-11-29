import numpy as np
import pandas as pd

class NaiveBayes(object):
	"""
	Although Naive Bayes is often introduced as one method,
	in fact it's one thinking, i.e. a framework of a series 
	of concrete methods. The common methods that based on 
	naive bayes framework include:

	- Boolean, i.e. only two classes, where P(Y) is Bernoulli
	  - Continuous P(X|Y)
	      1. Gaussian Naive Bayes, where P(X|Y) is Gaussian,
	         and P(Y) is Bernoulli.
	  - Discrete P(X|Y)
	      2. Multinomial Naive Bayes, where P(X|Y) is multinomial
	         and P(Y) is Bernoulli.
	      3. Bernoulli Naive Bayes, where P(X|Y) is Bernoulli,
	         and P(Y) is Bernoulli.

	- Multiclass, i.e. more than two classes, where P(Y) is 
	  Multinomial. And its subclass methods are the same as
	  above discussion.
	  - Continuous P(X|Y)
	      4. Gaussian Naive Bayes, where P(X|Y) is Gaussian,
	         and P(Y) is Multinomial.
	  - Discrete P(X|Y)
	      5. Multinomial Naive Bayes, where P(X|Y) is multinomial
	         and P(Y) is Multinomial.
	      6. Bernoulli Naive Bayes, where P(X|Y) is Bernoulli,
	         and P(Y) is Multinomial.

	Usually, e.g. in scikit-learn, the naive bayes implements only
	the part that P(Y) is Bernoulli. 

	"""

	def __init__(self):
		self.counts_table = None
		self.prob_table = {}

	def train(self, X, y):
		pass

	def predict(self, X):
		pass

	def loss(self):
		pass

class MultinomialNB(NaiveBayes):

	def train(self, X, y, laplaceS=0.0):
		"""
		Input
		-----
		- X: the observed data
		- y: the target data corresponding to y
		"""
		num_train, num_attrs = X.shape
		self.classes = np.unique(y)

		self.counts_table = pd.DataFrame(X)
		self.counts_table[num_attrs] = y
		df = self.counts_table

		for yVal in y:
			yValCounts = len(df[ df[num_attrs] == yVal ])
			self.prob_table[str(yVal)] = (yValCounts + 0.0) / num_train

			for X_field in X:
				for X_attr_indx, X_attr in enumerate(X_field):
					k_Xy = str(X_attr) + '|' + str(yVal)
					if not k_Xy in self.prob_table:
						Xy_counts = len(df[ (df[X_attr_indx] == X_attr) & (df[num_attrs] == yVal) ])
						self.prob_table[k_Xy] = (Xy_counts + laplaceS + 0.0) / (yValCounts + laplaceS * num_attrs)
						

	def predict(self, X):

		num_train, num_attrs = self.counts_table.shape
		num_attrs -= 1
		classes = self.classes
		
		for yVal in classes:
			y_pred = self.prob_table[str(yVal)]

			for xVal in X:
				query_str = str(xVal) + '|' + str(yVal)
				y_pred *= self.prob_table[query_str]

		res = classes[np.argmax(yVal)]
		print res
		