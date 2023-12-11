import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from src.convolution import n_convolutions, n_3d_convolutions
from src.cnn import CNNLayer

np.set_printoptions(suppress=True, linewidth=100000)

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
    for i in range(0, X.shape[0], batch_size):
        y_split = [y[i] for i in random_shuffle[i: i + batch_size]]
        yield X[random_shuffle[i: i + batch_size]], y_split


def part_two(Xtrain):
    kernels = [
        # np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]), # Edge detection
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]), # Sharpen
        1/9 * np.ones((3,3)), # box blur
    ]
    for k in kernels:
        print(k)

    timer = Timer()
    y_conv = list(n_3d_convolutions(Xtrain, kernels))
    timer.time("y_conv")
    conv_lay = CNNLayer(kernel_shapes=[k.shape for k in kernels], alpha=0.00000004) #0.000000001
    epoch = 0
    while epoch < 3:
        for X_batch, y_batch in batch_split(Xtrain, y_conv, 1000):
            y_bars = conv_lay.forward_propagations(X_batch)
            timer.time("forward")
            deltas = [np.zeros(k.shape) for k in kernels]
            for x, y_bar, y_real in zip(X_batch, y_bars, y_batch):
                dL_dy = [2 * (y_b - y_r) for y_b, y_r in zip(y_bar, y_real)]
                timer.time("dL_dy")
                dL_dw = conv_lay.backward_propagation(x, dL_dy) # Bottle neck
                timer.time("backward")
                deltas = [delta + dl for delta, dl in zip(deltas, dL_dw)]
                timer.time("deltas")
            conv_lay.update_weights([d/X_batch.shape[0] for d in deltas])
            timer.time("update")    

        epoch += 1
        for k in conv_lay.kernels:
            print(k)
    print(timer)

if __name__== "__main__":
    (Xtrain, ytrain), (Xtest,ytest) = tf.keras.datasets.mnist.load_data()
    # part_one(Xtrain[:5])
    part_two(Xtrain)
    # print(Xtrain.shape)
    pass