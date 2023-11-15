import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random, itertools, time, pickle

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
    print(alpha, batch_size, epochs, reg)
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
    correct_perc = []
    for epoch in range(epochs):
        batches = batch_split(yX, batch_size=batch_size)
        for batch in batches:
            status = (batch @ w) < 1
            direction = (status[:, None] * batch).sum(axis=0)
            w = (1 - alpha * reg) * w + alpha * direction
        
        correct_perc.append((1 - ((yX @ w) < 1).sum()/y_train.shape[0]).item())
    
    return w, correct_perc

# Creates a generator that gives a random batch with fixed size.
def batch_split(X, batch_size):
    random_shuffle = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    for i in range(0, X.shape[0], batch_size):
        yield X[random_shuffle[i: i + batch_size]]


def parameter_sampling(parameters):
    """
    Takes a dict like:
    {
        "parameter_1":[start, stop, step],
        "parameter_2":[start, stop, step],
        ...
    }
    And returns a cartesian product of all combinations of parameters.
    """
    uniform_sampling = [np.arange(*v).round(5).tolist() for v in parameters.values()]
    for r in itertools.product(*uniform_sampling):
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

def test_mini_batch_convergence(filename="batch_test_data"):
    """
    This samples some hyper parameters and records the performance, 
    then writes that away to a file, append only.
    """
    # alpha=0.001, batch_size=100, epochs=10, reg
    pars = parameter_sampling({
        "alpha":(0.0005, 0.0051, 0.001),
        "reg":(0.2, 1.20, 0.2),
        "batch_size%":(5, 30, 5)
    })
    epochs = 32

    X, y = parse_url(day=1)
    X = scipy.sparse.csr_array(X)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    yX_test = (X_test * y_test[:, None]).tocsr()

    for i, sample in enumerate(pars):
        t_start = time.time()
        weight, convergence = hinge_loss_bgd(
            X_train, 
            y_train, 
            alpha=sample.get("alpha", 0.0010), 
            batch_size=int(X_train.shape[0]*sample.get("batch_size%", 6)/100), 
            epochs=epochs, 
            reg=sample.get("reg", None)
            )

        t_end = time.time()
        result = {
                **sample,
                **{
                "convergence":convergence,
                "test_performance": (1 - ((yX_test @ weight) < 1).sum()/X_test.shape[0]).item(),
                "elapsed_time": t_end - t_start
                }
            }
        print(i, t_end - t_start)
        with open(filename, 'a+b') as fp:
            pickle.dump(result, fp)
    
if __name__ == "__main__":
    test_mini_batch_convergence()

    