import numpy as np
from convolution import n_convolutions, shape_indexs

class CNNLayer:
    def __init__(self, kernel_shapes) -> None:
        self.kernels = [np.zeros(shape) for shape in kernel_shapes]

    def forward_propagation(self, x):
        return n_convolutions(x, self.kernels, method="roll")

    def backward_propagation(self, x, ys):
        return n_convolutions(self, x, ys, method="roll")

                
        