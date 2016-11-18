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
		self.prob_table = None

	def train(self, x_domains, y_domains, X):
		
		pass

	def predict(self):
		"""
		Compute the probability based on the probability table. And
		choose the label with largest probability as its prediction.

		"""
		pass

	def loss(self):
		pass

class MultinomialNB(NaiveBayes):
	def train(self, x_domains, y_domains, X):
		"""
		This method accepts data in form [('X'), ('y')], where 'X'
		and 'y' may contain multiple attributes.

		The purpose of this method is to construct the probability
		table for each combination of 'X' and 'y'.


		Returns
        -------
        self : object
            Returns self.
		"""
		X_pd = []
		for r in X:
			flat_row = list(r[0])
			flat_row.append(r[1])
			X_pd.append(flat_row)

		self.prob_table = pd.DataFrame(X_pd)

		# df = self.prob_table
		# print df
		# print df[df[0] == '1']

