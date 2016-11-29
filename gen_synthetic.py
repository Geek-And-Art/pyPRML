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

def gen_naive_bayes_synthetic(X_domains, y_domains, N, hard_code=False):
    """
    Return an object which contains attributes: 'data', 'target'.
    Each element's attribute will be generated randomly from its domain.


    Inputs:
    - X_domains : A numpy array, in which the element represents one
                  inputting data's attribute's domain set.  
    - y_domains : A numpy array, in which the element represents one
                  label's attrubute's domain set.
    - N         : The number of data to be generated.
    - hard_code : If it's true, it'll return the embedded hard code data.


    Outputs:
    This method will return an object. Each element is constituded
    with observed data 'X' and its label 'y' tuple, i.e. [('X'), ('y')].

    Let's use the embedded hard code data as example.

    'X' contains two attribute 'x1, x2', , i.e. X = ('x1', 'x2'). 
    And their value ranges are {1, 2, 3} and {S, L, M} separately.

    The value range of 'y' is {-1, 1}.

    In summary, the return numpy array's element is [('x1', 'x2'), ('y1',)],
    whose value ranges are {1, 2, 3}, {S, L, M} and {-1, 1} separately.

    Notes:
    ------
    In order to make the data type consistent, the data type in both X and y
    will be string type. Because even the number, it's used as one simple to
    indicate different class, which is equivalent to a string.
    """
    res = {}

    if hard_code:
        x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        x2 = list("SMMSSSMMLLLMMLL")
        y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
        assert(len(x1) == len(x2) and len(x2) == len(y))

        num_train = len(x1)
        X = []
        for indx in xrange(num_train):
            X_field = []
            X_field.append(x1[indx])
            X_field.append(x2[indx])
            X.append(X_field)

        res['data'] = X
        res['target'] = y
        return res

    # Random generation
    num_train = N
    num_attrs = len(X_domains)
    X = [] # np.zeros((num_train, num_attrs))
    y = [] # np.zeros(num_train)

    for indx in xrange(num_train):
        X_field = []
        for xAttr in X_domains:
            X_field.append(np.random.choice(xAttr, 1)[0])
        X.append(X_field)

        y.append(np.random.choice(y_domains, 1)[0])

    res['data'] = X
    res['target'] = y

    return res
