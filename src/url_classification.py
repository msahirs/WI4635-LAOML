import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random, itertools

from iris_classification import timing


@timing
def parse_url(filename="data/url.mat", day =1):
    v = scipy.io.loadmat(filename)[f'Day{ day}']
    X = v['data'][0][0]
    y = np.array([( -1) ** v['labels'][0][0][k][0] for k in range(len(v['labels'][0][0]))])
    return X , y

@timing
def split_dataset(X, y, frac=0.5):
    """
    Scipy has not build in way to shuffle on sparse matrices, like numpy.
    So instead we pick random incides, and create a masking over the Sparse array.
    """
    split_choice = np.random.choice(y.shape[0], size= (int(y.shape[0] * frac),), replace=False)
    split_mask = np.zeros_like(y, dtype=bool)
    split_mask[split_choice] = 1
    return X[~split_mask], X[split_mask], y[~split_mask], y[split_mask]

@timing
def hinge_loss_gd(X_train, y_train, alpha=0.001, N=100, reg=1):
    """
    Hinge loss gradient descent for a SVM.
    Each iteration the status indicates the samples, which are classified wrong.
    This is used to mask the array before summing it for the direction.

    CSR seems to be the fastest of the scipy.sparse arrays.
    """
    w = np.zeros(X_train.shape[1])
    status = np.ones_like(y_train)
    yX = (X_train * y_train[:, None]).tocsr() #This only needs to be calculated once.

    num_wrong = []
    for i in range(N):
        direction = (status[:, None] * yX).sum(axis=0)
        w = (1 - alpha * reg) * w + alpha * direction
        status = (yX @ w) < 1
        if i%10==0:num_wrong.append(status.sum())

    return w, num_wrong

@timing
def hinge_loss_bgd(X_train, y_train, alpha=0.001, batch_size=100, epochs=10, reg=None):
    """
    Mini-batch hinge loss gradient descent for a SVM.
    Each epoch the data is split in random batches 
    and then the status is used in the same as in the version without bacthes.

    CSR seems to be the fastest of the scipy.sparse arrays.
    """
    if reg is None:
        reg = 1/epochs

    w = np.zeros(X_train.shape[1])
    yX = (X_train * y_train[:, None]).tocsr() #This only needs to be calculated once.
    num_wrong = []
    for epoch in range(epochs):
        batches = batch_split(yX, batch_size=batch_size)
        for batch in batches:
            status = (batch @ w) < 1
            direction = (status[:, None] * batch).sum(axis=0)
            w = (1 - alpha * reg) * w + alpha * direction
        
        num_wrong.append(((yX @ w) < 1).sum())
    return w, num_wrong

# Creates a generator that gives a random batch with fixed size.
def batch_split(X, batch_size):
    random_shuffle = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    for i in range(0, X.shape[0], batch_size):
        yield X[random_shuffle[i: i + batch_size]]


def parameter_sampling(parameters):
    random_choices = [[random.uniform(v[1], v[2]) for _ in range(v[0])] for v in parameters.values()]
    print(random_choices)
    for r in itertools.product(*random_choices):
        yield dict(zip(parameters.keys(), r))


def test_alpha_convergence():
    X, y = parse_url(day=1)
    X = scipy.sparse.csr_array(X)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    N = 100

    weights_hl, num_wrong_001 = hinge_loss_gd(X_train=X_train, y_train=y_train, alpha=0.001, N=N)
    weights_hl, num_wrong_0005 = hinge_loss_gd(X_train=X_train, y_train=y_train, alpha=0.0005, N=N)
    weights_hl, num_wrong_0015 = hinge_loss_gd(X_train=X_train, y_train=y_train, alpha=0.0015, N=N)

    plt.figure(figsize = (10, 8))

    plt.plot(range(0, N, 10), num_wrong_001,
            color = 'b',
            label = "alpha=0.001",
            marker = 'X', markersize = 3)
    
    plt.plot(range(0, N, 10), num_wrong_0005,
            color = 'r',
            label = "alpha=0.0005",
            marker = 'X', markersize = 3)
    
    plt.plot(range(0, N, 10), num_wrong_0015,
            color = 'orange',
            label = "alpha=0.0015",
            marker = 'X', markersize = 3)

    plt.xlabel('Number of rounds')
    plt.ylabel('Wrongly classified in train set')
    plt.legend()
    plt.savefig("alphagraph.png")

def test_mini_batch_convergence(plots = False): # Added visualisations

    pars = parameter_sampling({
        "alpha":(5, 0.0001, 0.005),
        "reg":(5, 0.01, 1),
    })
    
    X, y = parse_url(day=1)
    X = scipy.sparse.csr_array(X)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    epochs = 200

    weights_hl, num_wrong_001 = hinge_loss_bgd(X_train, y_train, alpha=0.001, batch_size=X_train.shape[0]//15, epochs=epochs, reg=None)
    weights_hl, num_wrong_0005 = hinge_loss_bgd(X_train, y_train, alpha=0.0005, batch_size=X_train.shape[0]//15, epochs=epochs, reg=None)
    weights_hl, num_wrong_0015 = hinge_loss_bgd(X_train, y_train, alpha=0.0015, batch_size=X_train.shape[0]//15, epochs=epochs, reg=None)
    weights_hl, num_wrong_002 = hinge_loss_bgd(X_train, y_train, alpha=0.002, batch_size=X_train.shape[0]//15, epochs=epochs, reg=None)
    
    # print(num_wrong_001)
    # print(num_wrong_0005)
    # print(num_wrong_0015)
    # print(num_wrong_002)

    if plots:

        plt.figure(figsize = (10, 8))

        plt.plot(range(0, epochs, 1), num_wrong_0005,
                color = 'r',
                label = "alpha=0.0005",
                marker = 'X', markersize = 3)

        plt.plot(range(0, epochs, 1), num_wrong_001,
                color = 'b',
                label = "alpha=0.001",
                marker = 'X', markersize = 3)
        

        plt.plot(range(0, epochs, 1), num_wrong_0015,
                color = 'orange',
                label = "alpha=0.0015",
                marker = 'X', markersize = 3)

        plt.plot(range(0, epochs, 1), num_wrong_002,
                color = 'g',
                label = "alpha=0.002",
                marker = 'X', markersize = 3)

        plt.xlabel('Number of Epochs')
        plt.ylabel('Wrongly classified in train set (mini-batch HL for SVM)')
        plt.legend()
        plt.savefig("alphagraph_mini.png")


if __name__ == "__main__":
    np.random.default_rng(1)
    # test_alpha_convergence()
    test_mini_batch_convergence(plots = True)
    