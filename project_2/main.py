import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
from itertools import product

from src.convolution import n_convolutions, n_3d_convolutions, rc, one_convolution
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
    kernel_names = ["Identity", "edge detection", "sharpen", "box blur", "gaussian blur"]
    kernels = [
        np.array([[0,0,0],[0,1,0],[0,0,0]]), # Identity
        np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]), # Edge detection
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]), # Sharpen
        1/9 * np.ones((3,3)), # box blur
        1/256 * np.outer(np.array([1,4,6,4,1]), np.array([1,4,6,4,1])), # Gaussian blur
    ]
    convolved_data = []
    for x in Xtrain:
        convolved_data.extend([one_convolution(x, k, method="loop") for k in kernels])
    f, axs = plt.subplots(5, 5)
    for i, ax in enumerate(axs.flatten()):
        sns.heatmap(
            data=convolved_data[i],
            vmin=0, 
            vmax=256,
            xticklabels=(5 if i>=20 else False), 
            yticklabels=(5 if i%5 == 0 else False),
            ax=ax,
            cmap="gray",
            cbar=False,
            square=True
        )
        if i < 5:
            ax.set_title(f"{kernel_names[i]}")
    f.suptitle("Multiple convolution kernels applied to 5 different inputs.")
    f.set_edgecolor("black")
    f.savefig("test.png")


def batch_split(X, y, batch_size):
    random_shuffle = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    for i in range(0, X.shape[0], batch_size):
        y_split = [y[i] for i in random_shuffle[i: i + batch_size]]
        yield X[random_shuffle[i: i + batch_size]], y_split

def part_two(Xtrain):
    epochs = 1
    normalize = 2

    if normalize == 2: # 01 normalize
        alpha = 0.3 # alpha 0.05
        batch_size = 10 # batch size 100, seems to converge fast

        Xtrain_min = Xtrain.min()
        Xtrain_max = Xtrain.max()
        Xtrain = (Xtrain - Xtrain_min)/(Xtrain_max - Xtrain_min)

    elif normalize ==1: # std normalize
        alpha = 0.0002 # alpha 0.05
        batch_size = 1000 # batch size 100, seems to converge fast

        Xtrain_mean = Xtrain.mean()
        Xtrain_std = np.std(Xtrain)
        Xtrain = (Xtrain - Xtrain_mean)/Xtrain_std

    else:
        alpha = 0.00000003 # alpha 0.0000005
        batch_size = 1000 # batch size 100, seems to converge fast
    
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
    # conv_lay = ConvLay(kernel_shapes=[k.shape for k in kernels], alpha=alpha, input_shape=Xtrain[0].shape) #0.000000001
    cnn_config = {
        "convolution": {"kernel_shapes":[(3,3)] * 3, "alpha":alpha},
    }
    cnn = ConvNN(cnn_config, Xtrain.shape[1:])

    cnn.train(
        Xtrain, 
        y_conv, 
        batch_size,
        loss_function="MSE",
        epochs=epochs
    )
    timer.time("epoch")

    print(f"after epoch")

    for k_b, k_r in zip(cnn.layers[0].kernels, kernels):
        print(k_b)
        print("max error", np.max(np.abs(k_b - k_r)))
    print("MSE", ((cnn.layers[0].kernels - kernels)**2).mean())
    print(timer)

def to_one_hot_vector(y):
    one_hot = np.zeros(10)
    one_hot[y] = 1
    return one_hot

def part_three(Xtrain, ytrain, Xtest, ytest):
    # Configs
    cnn_config = {
        "convolution": {"kernel_shapes":[(3,3)] * 3, "alpha":0.02},
        "min_max_pool": {"pool_shape":(2,2), "strides":(2, 2)},
        "soft_max":{"output_length":10, "alpha":0.001}
    }
    cnn = ConvNN(cnn_config, Xtrain.shape[1:])
    timer = Timer()

    epochs = 1
    batch_size = 50

    # Normalize
    Xtrain_min = Xtrain.min()
    Xtrain_max = Xtrain.max()
    Xtrain = (Xtrain - Xtrain_min)/(Xtrain_max - Xtrain_min)

    Xtest_min = Xtest.min()
    Xtest_max = Xtest.max()
    Xtest = (Xtest - Xtest_min)/(Xtest_max - Xtest_min)

    ytrain = list(map(to_one_hot_vector, ytrain))
    ytest = list(map(to_one_hot_vector, ytest))

    for i in range(10):
        cnn.train(
            Xtrain,
            ytrain,
            batch_size,
            loss_function="cross_entropy",
            epochs=1
        )

        model_weights = cnn.save()

        with open("models", 'a+b') as fp:
            pickle.dump(model_weights, fp)
        print(model_weights)
        # y_bars = cnn.forward_propagations(Xtest)

        # losses = map(
        #     cnn.log_loss,
        #     y_bars, ytest
        # )
        # print(sum(losses))
        print(timer)
        print(rc)
        break

def k_cross_validate(Xtrain, ytrain):
    Xtrain_min = Xtrain.min()
    Xtrain_max = Xtrain.max()
    Xtrain = (Xtrain - Xtrain_min)/(Xtrain_max - Xtrain_min)
    ytrain = list(map(to_one_hot_vector, ytrain))

    k_split = list(batch_split(Xtrain, ytrain, 20000))

    options = [
        [(3,3), (5,5)],
        [3,5],
        [0.02, 0.01, 0.005],
        [0.002, 0.001, 0.0005]
    ]

    for comb, (shape, num, alpha_k, alpha_s) in enumerate(product(*options)):
        print(comb, shape, num, alpha_k, alpha_s)
        cnn_config = {
            "convolution": {"kernel_shapes":[shape] * num, "alpha":alpha_k},
            "min_max_pool": {"pool_shape":(2,2), "strides":(2, 2)},
            "soft_max":{"output_length":10, "alpha":alpha_s}
        }
        cnn = ConvNN(cnn_config, Xtrain.shape[1:])
        losses = []
        for k in range(3):
            timer = Timer()
            k_Xtrain = []
            k_ytrain = []
            for i, ks in enumerate(k_split):
                if i == k:
                    continue
                k_ytrain.extend(ks[1])
                k_Xtrain.append(ks[0])
            k_Xtrain = np.concatenate(k_Xtrain)
            print(k_Xtrain.shape)
            timer.time("setup")
            print("training")
            cnn.train(
                k_Xtrain,
                k_ytrain,
                batch_size=50,
                loss_function="cross_entropy"
            )
            timer.time("train")
            print("predicting")
            y_bars = list(cnn.forward_propagations(k_split[k][0]))
            timer.time("predict")

            loss = sum(map(
                cnn.log_loss,
                y_bars, k_split[k][1]
            ))
            timer.time("loss")
            print(loss)
            print(timer)
            print(rc)
            losses.append(loss)
        result = {
            "shape":shape, 
            "number":num, 
            "kernel_alpha": alpha_k, 
            "softmax_alpha":alpha_s,
            "losses":losses
        }
        print(result)
        with open("k-fold_test", 'a+b') as fp:
            pickle.dump(result, fp)

def test_case():
    data = []
    with open("models", 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    for d in data:
        print(d)
    return data


if __name__== "__main__":
    (Xtrain, ytrain), (Xtest,ytest) = tf.keras.datasets.mnist.load_data()
    # part_one(Xtrain[:5])
    # part_two(Xtrain)
    part_three(Xtrain, ytrain, Xtest, ytest)
    # k_cross_validate(Xtrain, ytrain)
    test_case()
    # print(rc)
    # print(Xtrain.shape)
