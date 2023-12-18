import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from src.convolution import n_convolutions, n_3d_convolutions, rc
from src.cnn import ConvNN, ConvLay

# np.set_printoptions(suppress=True, linewidth=100000)

class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.timings = {}
    
    def time(self, part):
        self.timings[part] = self.timings.get(part, 0) + time.time() - self.start_time
        self.start_time = time.time()
    
    def __str__(self):
        return self.timings.__str__()

def part_one(Xtrain):
    kernels = [
        np.array([[0,0,0],[0,1,0],[0,0,0]]), # Identity
        np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]), # Edge detection
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]), # Sharpen
        1/9 * np.ones((3,3)), # box blur
        1/256 * np.outer(np.array([1,4,6,4,1]), np.array([1,4,6,4,1])), # Gaussian blur
    ]
    convolved_data = []
    for x in Xtrain:
        convolved_data.extend(n_convolutions(x, kernels, method="roll"))
    
    f, axs = plt.subplots(5, 5)
    for i, ax in enumerate(axs.flatten()):
        sns.heatmap(
            data=convolved_data[i],
            vmin=0, 
            vmax=256,
            xticklabels=False, 
            yticklabels=False,
            ax=ax,
            cmap="gray",
            cbar=False,
            square=True
        )
    f.savefig("test.png")


def batch_split(X, y, batch_size):
    random_shuffle = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    # random_shuffle = np.arange(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        y_split = [y[i] for i in random_shuffle[i: i + batch_size]]
        yield X[random_shuffle[i: i + batch_size]], y_split

def mse_gradient(y_bars, ys):
    """
        Returns the gradient of MSE
    """
    # size = sum([np.prod(y.shape) for y in ys])
    # return [2 * (y_b - y)/size for y_b, y in zip(y_bars, ys)]
    return 2 * (y_bars - ys)/y_bars.size 

def part_two(Xtrain):
    epochs = 1
    normalize = 1

    if normalize == 2: # 01 normalize
        alpha = 0.005 # alpha 0.05
        batch_size = 100 # batch size 100, seems to converge fast

        Xtrain_min = Xtrain.min()
        Xtrain_max = Xtrain.max()
        Xtrain = (Xtrain - Xtrain_min)/(Xtrain_max - Xtrain_min)

    elif normalize ==1: # std normalize
        alpha = 0.003 # alpha 0.05
        batch_size = 100 # batch size 100, seems to converge fast

        Xtrain_mean = Xtrain.mean()
        Xtrain_std = np.std(Xtrain)
        Xtrain = (Xtrain - Xtrain_mean)/Xtrain_std

    else:
        alpha = 0.0000005 # alpha 0.0000005
        batch_size = 100 # batch size 100, seems to converge fast
    
    kernels = np.stack([
        np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]), # Edge detection
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]), # Sharpen
        1/9 * np.ones((3,3)), # box blur
    ])
    print("starting kernels")
    print(kernels)
    timer = Timer()
    y_conv = list(n_3d_convolutions(Xtrain, kernels))

    timer.time("y_conv")
    conv_lay = ConvLay(kernel_shapes=[k.shape for k in kernels], alpha=alpha) #0.000000001
    epoch = 0

    while epoch < epochs:
        for X_batch, y_batch in batch_split(Xtrain, y_conv, batch_size):
            y_bars = conv_lay.forward_propagations(X_batch) # Does not evaluate yet.
            timer.time("forward")
            dL_dys = map(
                mse_gradient,
                y_bars, y_batch
                ) # This creates an iterator, so it stays lazy
            timer.time("dl_dy")
            conv_lay.backward_propagations(dL_dys)
            timer.time("backward")

        print(f"after epoch {epoch}")
        for k_b, k_r in zip(conv_lay.kernels, kernels):
            print(k_b)
            print("max error", np.max(np.abs(k_b - k_r)))
        print()
        epoch += 1

    print(timer)

def part_three(Xtrain):
    cnn_config = {
        "convolution": {"kernel_shapes":[(3,3), (3,3)], "alpha":0.0005},
        "min_max_pool": {},
        "soft_max":{}
    }
    cnn = ConvNN(cnn_config)
    y_bar = cnn.forward_propagations(Xtrain)
    cnn.backward_propagations(y_bar)

if __name__== "__main__":
    (Xtrain, ytrain), (Xtest,ytest) = tf.keras.datasets.mnist.load_data()
    # Xtrain = np.array([[[0,1,1,0],[1,1,1,0],[0,1,0,1],[1,0,0,1]]])
    # part_one(Xtrain[:5])
    part_two(Xtrain)
    # part_three(Xtrain[:100])
    print(rc)
    # print(Xtrain.shape)
