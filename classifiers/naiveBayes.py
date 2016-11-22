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

	def train(self, x_domains, y_domains, X):
		
		pass

	def predict(self):
		pass

	def loss(self):
		pass

class MultinomialNB(NaiveBayes):

	def train(self, x_domains, y_domains, dataPkg, laplaceS=0.0):
		"""
		This method accepts data in form [('X'), ('y')], where 'X'
		and 'y' may contain multiple attributes.

		The purpose of this method is to construct the probability
		table for each combination of 'X' and 'y'.

		The probability table is a dictionary. It stores in two forms:
		- self.prob_table['y'] = the ratio of value 'y' among whole data
		- self.prob_table['x|y'] = the conditional probability 'x|y'
		"""
		self.x_domains = x_domains
		self.y_domains = y_domains

		X_pd = []
		for r in dataPkg:
			flat_row = list(r[0])
			flat_row.append(r[1])
			X_pd.append(flat_row)

		self.counts_table = pd.DataFrame(X_pd)
		df = self.counts_table

		for yDom in y_domains:
			for yVal in yDom:
				yValCounts = len(df[df[len(x_domains)] == yVal])
				dfCounts = len(df)
				self.prob_table[yVal] = (yValCounts + 0.0) / dfCounts

				for i, xDom in enumerate(x_domains):
					for xVal in xDom:
						xyValCounts = len(df[(df[i] == xVal) & (df[len(x_domains)] == yVal)])
						xFeatureCounts = len(xDom)
						self.prob_table[xVal + '|' + yVal] = (xyValCounts + laplaceS + 0.0) / (yValCounts + laplaceS * xFeatureCounts)


	def predict(self, X):
		assert(len(X) == len(self.x_domains))

		y_pred = np.ones_like(self.y_domains, dtype=float)

		for j, yDom in enumerate(self.y_domains):
			for i, yVal in enumerate(yDom):
				y_pred[j][i] = self.prob_table[yVal]

				for xVal in X:
					query_str = xVal + '|' + yVal

					# Base on naive bayes assumption.
					y_pred[j][i] *= self.prob_table[query_str]

		res = []
		for i, y in enumerate(y_pred):
			res.append(self.y_domains[i][np.argmax(y)])

		print res
		