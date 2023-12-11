import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.convolution import n_convolutions


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

def part_two():
    pass

if __name__== "__main__":
    (Xtrain, ytrain), (Xtest,ytest) = tf.keras.datasets.mnist.load_data()
    part_one(Xtrain[:5])
    print(Xtrain.shape)
    pass