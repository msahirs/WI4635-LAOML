import numpy as np
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


iris_data_loc = './data/iris.csv' # File name of IRIS dataset

data_array = extract_dataset(iris_data_loc)

train_X, train_y, test_X, test_y = split_data_rnd(data_array, fraction=0.5)

no_correct, results_sign, weight_vec = cluster_dataset_test(train_X, train_y, test_X, test_y)




