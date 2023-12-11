import numpy as np
from .convolution import n_convolutions, one_convolution, n_3d_convolutions

class CNNLayer:
    def __init__(self, kernel_shapes, alpha) -> None:
        nodes = sum((np.prod(s) for s in kernel_shapes))
        self.kernels = [(np.random.rand(*shape) - 1/2)/np.sqrt(nodes) for shape in kernel_shapes]
        self.alpha = alpha

    def forward_propagation(self, x):
        return n_convolutions(x, self.kernels)
    
    def forward_propagations(self, xs):
        return n_3d_convolutions(xs, self.kernels)

    def backward_propagation(self, x, ys, dx=False):
        """
            We use the the partial derivatives of the
            cost function wrt to the weights to update
            the weights.
            And we return the derivatices of the cost
            function wrt to input.
        """
        dL_dw = n_convolutions(x, ys)
        if dx:
            dL_dx = sum([
                one_convolution(w, np.flip(y), padding=True) 
                for w, y in zip(self.kernels, ys)
                ])
            return dL_dw, dL_dx
        else:
            return dL_dw

    def update_weights(self, ups):
        self.kernels = [
            k - self.alpha * up
            for k, up 
            in zip(self.kernels, ups)
            ]