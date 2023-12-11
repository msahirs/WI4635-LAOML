import numpy as np
from .convolution import n_convolutions, one_convolution

class CNNLayer:
    def __init__(self, kernel_shapes, alpha) -> None:
        nodes = sum((np.prod(s) for s in kernel_shapes))
        self.kernels = [(np.random.rand(*shape) - 1/2)/np.sqrt(nodes) for shape in kernel_shapes]
        for k in self.kernels:
            print(k)
        self.alpha = alpha

    def forward_propagation(self, x):
        return n_convolutions(x, self.kernels)

    def backward_propagation(self, x, ys):
        """
            We use the the partial derivatives of the
            cost function wrt to the weights to update
            the weights.
            And we return the derivatices of the cost
            function wrt to input.
        """
        self.kernels = [k + self.alpha * up
            for k, up 
            in zip(
                self.kernels, 
                n_convolutions(self, x, ys)
                )
            ]
        
        return sum([one_convolution(w, np.flip(y), padding=True) for w, y in zip(self.kernels, ys)])
                
        