import numpy as np
from numpy.linalg import qr
import scipy as sc
import matplotlib.pyplot as plt
import csv

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

    if iris_data_loc.split(".")[-1] != "csv":
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

    return np.hstack((sample_data, outcome_vec[:,np.newaxis]))

def split_data_rnd(dataset,fraction = 0.5):

    train_size = int(dataset.shape[0] * fraction)
    # test_size = dataset.shape[0] - train_size
    
    # shuffled_data = np.copy(dataset)

    np.random.shuffle(dataset)

    train_X , train_y = dataset[:train_size,:-1], dataset[:train_size,-1],
    test_X , test_y = dataset[train_size:,:-1], dataset[train_size:,-1],  

    return train_X, train_y, test_X, test_y

def train_clustering(train_data, train_outcome):

    return np.linalg.solve(train_data.T @ train_data, train_data.T @ train_outcome)

def cluster_dataset_test(train_X, train_y, test_X, test_y):
    weight_vec = train_clustering(train_X, train_y)

    results = test_X @ weight_vec

    results_sign = np.sign(results)

    no_correct = {np.sum(results_sign * test_y >0)}

    return no_correct, results_sign, weight_vec

def tikhonov_qr_lse(A,b,reg_param = 1):
    n_param = A.shape[1]

    A_reg = np.vstack((A,np.sqrt(reg_param)*np.eye(n_param)))
    b_reg = np.concatenate((b,np.zeros(n_param)))
    
    q, r = qr(A_reg)

    rhs = q.T @ b_reg

    return np.linalg.solve(r,rhs)


def test_1():
    # generate x and y
    x = np.linspace(0, 1, 101)
    y = 1 + x + x * np.random.random(len(x))

    # assemble matrix A
    A = np.vstack([x, np.ones(len(x))]).T

    tik = tikhonov_qr_lse(A,y,reg_param=1)

    print(tik)

    # turn y into a column vector
    y = y[:, np.newaxis]
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
    print(alpha)    

    # plot the results
    plt.figure(figsize = (10,8))
    plt.plot(x, y, 'b.')
    plt.plot(x, alpha[0]*x + alpha[1], 'r')
    plt.plot(x, tik[0]*x + tik[1], 'g')


    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


iris_data_loc = './data/iris.csv' # File name of IRIS dataset

data_array = extract_dataset(iris_data_loc)

train_X, train_y, test_X, test_y = split_data_rnd(data_array, fraction=0.5)

no_correct, results_sign, weight_vec = cluster_dataset_test(train_X, train_y, test_X, test_y)

test_1()






