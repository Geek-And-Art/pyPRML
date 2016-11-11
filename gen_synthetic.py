import numpy as np


def gen_synthetic(N, D, sigma):
    """
    Generate synthetic dataset based on given parameters.

    First, it'll generate a numpy array dataset 'X' of shape (N, D), in 
    which the elements governed by normal distribution N(0, 1).

    Then, it'll generate a weights numpy array 'W' of shape (D, 1), which 
    is governed by uniform distribution U(-0.5, 0.5).

    Finally, X's corresponding class dataset 'Y' of shape(N, 1), will be
    generated. It's computed through scores, which is the multiplication 
    of X and W, plus a system error, which is governed by distribution 
    N(0, sigma). Then, based on the fact that if element of scores is less 
    than 0, the corresponding Y element will be assigned with -1 or 1.

    Inputs:
    - N    : The number of rows of the generated numpy array 'X'.
    - D    : The number of columns of the generated numpy array 'X'.
    - sigma: The standard deviation of normal distribution, which
             governs the distribution of system error adding to the 
             multiplication of X and W.

    Returns a list of:
    - X: a numpy array of shape(N, D) under distribution N(0, 1)
    - Y: a numpy array of shape(N, 1) with '-1' or '1' element.
    """
    
    X = np.random.normal(size=(N, D))
    print "X has been generated."
    
    W = np.random.uniform(size=(D, 1)) - 0.5
    print "W has been generated."    
    
    Y0 = np.dot(X, W) 
    Y0 = Y0 + np.random.normal(loc=0, scale=sigma, size=(N, 1))
    
    Y = np.ones((N, 1))
    Y[Y0 < 0] = -1
    Y[Y0 >= 0] = 1
    print "Y has been generated."
    
    return X, Y
    