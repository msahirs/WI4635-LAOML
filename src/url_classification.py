import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from iris_classification import timing


@timing
def parse_url(filename="data/url.mat", day =1):
    v = scipy.io.loadmat(filename)[f'Day{ day}']
    X = v['data'][0][0]
    y = np.array([( -1) ** v['labels'][0][0][k][0] for k in range(len(v['labels'][0][0]))])
    return X , y

@timing
def split_dataset(X, y, frac=0.5):
    rng = np.random.default_rng(1)
    split_choice = rng.choice(y.shape[0], size= (int(y.shape[0] * frac),), replace=False)
    split_mask = np.zeros_like(y, dtype=bool)
    split_mask[split_choice] = 1
    return X[~split_mask], X[split_mask], y[~split_mask], y[split_mask]

@timing
def hinge_loss_gd(X_train, y_train, alpha=0.001, N=100, reg=1):
    w = np.zeros(X_train.shape[1])
    status = np.ones_like(y_train)
    yX = X_train * y_train[:, None] #This only needs to be calculated once.
    num_wrong = []
    for i in range(N):
        direction = (status[:, None] * yX).sum(axis=0)
        w = (1 - alpha * reg) * w + alpha * direction
        status = (yX @ w) < 1
        if i%10==0:num_wrong.append(status.sum())

    return w, num_wrong

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

if __name__ == "__main__":
    test_alpha_convergence()
    