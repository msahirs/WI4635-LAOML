import numpy as np
from numpy.linalg import qr
import matplotlib.pyplot as plt
import csv

from functools import wraps
from timeit import default_timer

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = default_timer()
        result = f(*args, **kw)
        te = default_timer()
        print('func:%r took: %.8f ms' % \
          (f.__name__, (te-ts) * 1000))
        return result
    return wrap


# Importing csv module

def extract_dataset(data_file : str) -> np.ndarray: 

    """ Partitions IRIS data in CSV format into its data matrix and outcome vector

    Parameters
    ----------
    data_file : str
        Relative path to IRIS dataset in .csv format

    Returns
    ----------
        NDArray
            Data matrix in cols [:-1], Outcome vector in col [-1]
    """

    if data_file.split(".")[-1] != "csv":
        raise ValueError("Not a .csv file. Exiting...")
    
    # Read CSV file and store into list of lists
    with open(data_file, "r") as x:
        sample_data = list(csv.reader(x, delimiter=","))

    headers = sample_data[0] # Store headers for convenience
    sample_data.pop(0) # Remove header list

    # versicolor gets assigned 1 and virginica -1
    outcome_vec = np.array([1 if i[-1]== "versicolor" else -1 for i in sample_data])

    # remove labels
    for i in sample_data:
        i.pop()

    # convert list of lists to ndarray of floats
    sample_data = np.array(sample_data, dtype=float)

    # Returns as single nd array with last column as outcome
    return np.hstack((sample_data, outcome_vec[:,np.newaxis]))

def split_data_rnd(dataset, fraction = 0.5):

    """ Shuffles dataset and splits according to specified fraction

    Parameters
    ----------
    dataset : NDArray
        Array to shuffle. Array is shuffled along its first dimension, i.e.
        rows are shuffled if input array is 2-D

    fraction : float
        Fraction of shuffled dataset to return as training data. Remaining is returned as testing

    Returns
    ----------
        train_X : NDArray
            Training data array, 2-D

        train_y : NDArray
            Training outcome vector, 1-D

        test_X : NDArray
            Testing data array, 2-D

        test_y : NDArray
            Testing outcome vector, 1-D
    """    

    train_size = int(dataset.shape[0] * fraction) # Convert fraction to no. of elements of training data
    
    # shuffled_data = np.copy(dataset) # realised not needed cause python is call by value lol

    np.random.shuffle(dataset) # N.B. In-place function!

    # split shuffled data accoridng to fraction
    train_X , train_y = dataset[:train_size,:-1], dataset[:train_size,-1],
    test_X , test_y = dataset[train_size:,:-1], dataset[train_size:,-1],  

    return train_X, train_y, test_X, test_y

def cluster_dataset_test(train_X, train_y, test_X, test_y):
    """Uses linear regression to build a hyperplane based on input training data.
    Provided test data and its outcome used as a means of testing and validation.

    Parameters
    ----------
    train_X : NDArray
        Training data array, 2-D

    train_y : NDArray
        Training outcome vector, 1-D

    test_X : NDArray
        Testing data array, 2-D

    test_y : NDArray
        Testing outcome vector, 1-D

    Returns
    ----------
    no_correct : int
        Number of correctly identified test entries

    results_sign : NDArray
        Vector containing classification output  

    weight_vec : NDArray
        Vector containing weights to define hyperplane
    """    

    # Solve least-squares weights in the most generic way using matrix_transposed-to-matrix product
    @timing
    def train_clustering(train_data, train_outcome):
        return np.linalg.solve(train_data.T @ train_data, train_data.T @ train_outcome)
    
    # Call private lse function to get weights
    weight_vec = train_clustering(train_X, train_y)

    # Get raw outcome vector
    results = test_X @ weight_vec

    # Binary normalisation
    results_sign = np.sign(results)

    # Number of correct is sum of the product ground truth signs and estimated signs
    no_correct = {np.sum(results_sign * test_y >0)}
    
    return no_correct, results_sign, weight_vec

@timing  
def generic_lse(train_data, train_outcome): # Does not use matrix inverse, but direct linear system solve
        return np.linalg.solve(train_data.T @ train_data, train_data.T @ train_outcome)

@timing    
def tikhonov_qr_lse(A, b,reg_param = 1):

    n_param = A.shape[1] # Get number of parameters

    # Tikhonov regularisation for lse can essentially be implemented appending
    # more rows to represent the L-2 norm term (per parameter)
    A_reg = np.vstack((A, np.sqrt(reg_param) * np.eye(n_param)))
    b_reg = np.concatenate((b, np.zeros(n_param)))
    
    # Use lib routine from numpy to get QR factorisation
    q, r = qr(A_reg)

    # Compute RHS of lse problem via QR
    rhs = q.T @ b_reg

    # Solve and return solution of lse via qr
    return np.linalg.solve(r, rhs)

def linear_CG(A, b, x = None, epsilon = 1e-8):
   
   if x is None:
    x = np.ones(b.size)
   
    res = A.dot(x) - b # Initialize the residual
    delta = -res # Initialize the descent direction
    
    while True:
        
        if np.linalg.norm(res) <= epsilon:
            return x # Return the minimizer x*
        
        D = A.dot(delta)
        beta = -(res.dot(delta))/(delta.dot(D)) 
        x = x + beta*delta # Generate the new iterate

        res = A.dot(x) - b # generate the new residual
        chi = res.dot(D)/(delta.dot(D)) 
        delta = chi*delta -  res # Generate the new descent direction

@timing
def tikhonov_cg_lse(data, outcome, reg_param = 1, x = None, epsilon = 1e-8):

    n_param = data.shape[1] # Get number of parameters

    # Tikhonov regularisation for lse can essentially be implemented appending
    # more rows to represent the L-2 norm term (per parameter)
    A_reg = np.vstack((data, np.sqrt(reg_param) * np.eye(n_param)))
    b_reg = np.concatenate((outcome, np.zeros(n_param)))

    lse_lhs = A_reg.T @ A_reg
    lse_rhs = A_reg.T @ b_reg

    return linear_CG(lse_lhs, lse_rhs, x = x, epsilon = epsilon)

@timing
def _test_1(plot = True): # Genreic lse vs qr factorisation + tikhonov vs CG + tikhonov test
    # generate data and outcome

    N_data_pts = 100000

    data = np.linspace(0, 1, N_data_pts)
    outcome = 10 + data + data * np.random.random(data.size)

    # assemble matrix A
    A = np.vstack([data, np.ones((data.size))]).T

    reg_param = 1e-0 # Set prefactor for norm term in tikhonov reguularisation

    # Use QR + Tikhonov Reg function
    tik_qr_weights = tikhonov_qr_lse(A, outcome, reg_param = reg_param)

    # Use generic LSE function
    generic_weights = generic_lse(A, outcome)
    # print(alpha)    

    # Use CG + Tikhonov Reg function
    tik_cg_weights = tikhonov_cg_lse(A, outcome, reg_param = reg_param)

    # plot the results
    if plot == True:
    
        plt.figure(figsize = (10, 8))

        plt.scatter(data, outcome,
                color= 'b', alpha = 0.025,
                label = "Input Data",
                marker = '.')  # Raw data plot
        
        plt.plot(data, generic_weights[0]*data + generic_weights[1],
                color = 'orange',
                label = "Traditional LSE",
                marker = 'X', markersize = 3) # lse using direct solve (w/o inversion)

        plt.plot(data, tik_qr_weights[0]*data + tik_qr_weights[1],
                'g',
                label = f"QR LSE w/ Tikhonov ($\lambda = {reg_param}$)",
                marker = 'o', markersize = 3) # using qr factorisation w/ tikhonov l2 regularisation
        
        plt.plot(data, tik_cg_weights[0]*data + tik_cg_weights[1],
                color = 'black', alpha = 0.5,
                label = f"CG LSE w/ Tikhonov ($\lambda = {reg_param}$)",
                marker = '*', markersize = 5,
                linestyle = "--") # using qr factorisation w/ tikhonov l2 regularisation
        
        plt.xlabel('Input data $x$')
        plt.ylabel('Outcome $y$')
        plt.legend()
        plt.show()

def _test_2(): # test to check CG function for solving system of equations
    A = np.array([[-7, 1],
                  [3, -3]])
    
    b = np.array([2,3])

    print(linear_CG(A,b))

def _test_3(): # TODO: test for matrices of high condition numbers. Good check for regularisation
    pass

def main():

    iris_data_loc = './data/iris.csv' # File name of IRIS dataset

    data_array = extract_dataset(iris_data_loc)

    train_X, train_y, test_X, test_y = split_data_rnd(data_array, fraction=0.5)

    no_correct, results_sign, weight_vec = cluster_dataset_test(train_X, train_y, test_X, test_y)

    reg = 1e-0
    w_cg = tikhonov_cg_lse(train_X,train_y, reg_param= reg)
    w_qr = tikhonov_qr_lse(train_X,train_y, reg_param= reg)

    print("Weights of cg:", w_cg)
    print("Weights of qr:", w_qr)
    print("Weights of hyperplane clustering:", weight_vec)

    x = [i for i in range(w_cg.size)]
    plt.plot(x, w_cg ,label = "cg", marker = 'x', alpha = 0.8, linestyle = '--')
    plt.plot(x, w_qr, label = "qr", marker = 'o', alpha = 0.8,linestyle = '--')
    plt.plot(x, weight_vec, label = "hyperplane", marker = '^', alpha = 0.8,linestyle = '--')
    plt.legend()
    plt.show()

    # _test_1(plot = False)
    # _test_2()


if __name__ == "__main__":
    main()







