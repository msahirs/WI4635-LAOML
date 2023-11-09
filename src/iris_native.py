import csv, random, math
import numpy as np

from iris_classification import timing
from pymatrix import *

@timing
def unpack_csv(file_name='./data/iris.csv'):
    """
    Unpacks the CSV into python lists, where the flower name is transformed into 1 and -1.
    """
    X, y = [], []
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        columns = next(reader)
        for line in reader:
            X.append(list(map(float, line[:-1])))
            y.append(1 if line[-1]=="versicolor" else -1)

    return X, y, columns

@timing
def split_and_transform_data(X, y, fraction, seed=True):
    """
    Splits the data into 2 parts, where the fraction gives the test set size.
    And then transforms the data into DMatrices and DVec.
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    if seed:
        random.seed(35)
    
    split_indices = sorted(random.sample(range(len(X)), int(fraction * len(X))))
    # split_indices = list(range(0,100,2))
    for index, (X_i, y_i) in enumerate(zip(X, y)):
        if index in split_indices:
            X_train.append(X_i)
            y_train.append(y_i)
        else:
            X_test.append(X_i)
            y_test.append(y_i)
    return DMatrix(X_train), DMatrix(X_test), DVec(y_train), DVec(y_test)

@timing
def count_correct(X, y, w):
    y_pred = X @ w
    return (y_pred * y > 0).sum()


def weights_with_gd(X_train, y_train, reg_param, alpha=.0005):
    # We need to do these calculations only once
    X_gram = X_train.T @ X_train + reg_param * DMatrix.eye(X_train.shape[1])
    Xy = X_train.T @ y_train
    y_2 = y_train.T @ y_train
    w_start = DVec([1] * X_train.shape[1])

    def gradient_f(w):
        return X_gram @ w - Xy
    
    def score_f(w):
        return w.T @ X_gram @ w - w.T @ Xy - Xy.T @ w + y_2 + w.T @ w
    
    w_fixed, t = fixed_gd(grad_f=gradient_f, w=w_start, alpha=alpha, epsilon=1e-8)
    print(w_fixed, t)
    print(line_search(A=X_gram, b=Xy, w=w_start))
    return w_fixed, t

@timing
def fixed_gd(grad_f, w, alpha=0.001, epsilon = 1e-8, max_iter=50_000): #gd_fixed(grad_f, w, N, alpha, cut_off=1e-9):
    for t in range(max_iter):
        a_t = grad_f(w)
        w = w - alpha * a_t
        if math.sqrt((a_t.T @ a_t)) <= epsilon:
            break
    return w, t

@timing
def line_search(A, b, w, epsilon=1e-8):
    res = A @ w - b
    delta = -1 * res
    while True:
        if math.sqrt((res.T @ res)) <= epsilon:
            break

        D = A @ delta
        beta = -1 * (res.T @ delta)/(delta.T @ D) 
        w = w + beta*delta # Generate the new iterate

        res = A @ w - b # generate the new residual
        chi = (res.T @ D)/(delta.T @ D) 
        delta = chi*delta -  res # Generate the new descent direction
    return w

@timing
def perceptron(X_train, y_train, M_iter=500):

    x = DVec([1] * X_train.shape[1])
    i = 0
    status = (X_train @ x * y_train <= 0)
    while any(status) and i < M_iter:
        for idx, _ in filter(lambda x: x[1], enumerate(status)):
            x = x + y_train[idx] * X_train[idx]

    #     # x = x / np.linalg.norm(x)
        status = (X_train @ x.T * y_train <= 0)
        i += 1

    if i == M_iter:
        print("Did not exit early, data might not be linearly seperable")
    return x

if __name__ == "__main__":
    X, y, columns = unpack_csv()
    fraction = 0.2
    X_train, X_test, y_train, y_test = split_and_transform_data(X, y, fraction=fraction, seed=False)
    # w, t = weights_with_gd(X_train, y_train, reg_param=1, alpha=0.0005)
    # print(count_correct(X_test, y_test, w))
    perceptron(X_train=X_train, y_train=y_train)