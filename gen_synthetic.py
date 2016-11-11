import numpy as np


def gen_linear_synthetic(N, D, sigma):
    """
    Generate synthetic linear model dataset based on given parameters. 

    Its basic idea comes from the linear combination of 
        'x_0 * w0 + x_1 * w_1 + ... + x_D * w_D'
    where 'D' is the dimension of the inputting data. Then, add one 
    systematic error 'e' to this linear combination to express observed 
    data:
        '(x_0 * w0 + x_1 * w_1 + ... + x_D * w_D) + e'
    
    This is the process of just one data point generation. Our goal is 
    to generate N data points in this form with same weigh vector W. 
    Because we assume all the data points are governed by same model, 
    i.e. the same weight vector W. Thus, the matrix form of above formula 
    can be expressed as
        'X * W + E'
    where the X's shape is (N, D), W's shape is (D, 1), and E's shape is
    (N, 1).


    The detailed specifications of this process are:

    - First, it'll generate a numpy array dataset 'X' of shape (N, D), in 
      which the elements governed by normal distribution N(0, 1).

    - Then, it'll generate a weights numpy array 'W' of shape (D, 1), which 
      is governed by uniform distribution U(-0.5, 0.5).

    - Finally, X's corresponding class dataset 'Y' of shape(N, 1) will be
      generated. It's computed through scores, which is the multiplication 
      of X and W, plus a systematic error, which is governed by distribution 
      N(0, sigma). Then, based on the fact that if element of scores is less 
      than 0, the corresponding Y element will be assigned with -1 or 1.

    
    Inputs:
    - N    : The number of generated data points. Number of rows of the
             numpy array 'X'.
    - D    : The dimension of generated data point. Number of columns of
             the numpy array 'X'.
    - sigma: The standard deviation of normal distribution, which
             governs the distribution of systematic error added to the 
             multiplication of X and W.

    Returns a list of:
    - X: a numpy array of shape(N, D) under distribution N(0, 1)
    - Y: a numpy array of shape(N, 1) with '-1' or '1' element.
    """
    
    X = np.random.normal(size=(N, D))
    print "X has been generated."
    
    W = np.random.uniform(size=(D, 1)) - 0.5
    print "W has been generated."

    err_sys = np.random.normal(loc=0, scale=sigma, size=(N, 1))
    
    Y0 = np.dot(X, W) + err_sys
    
    Y = np.ones((N, 1))
    Y[Y0 < 0] = -1
    Y[Y0 >= 0] = 1
    print "Y has been generated."
    
    return X, Y
