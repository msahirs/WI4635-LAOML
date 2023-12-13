import numpy as np
from .convolution import n_convolutions, one_convolution, n_3d_convolutions

class CNNLayer:
    def __init__(self, kernel_shapes, alpha) -> None:
        nodes = sum((np.prod(s) for s in kernel_shapes))
        self.kernels = [(np.random.rand(*shape) - 1/2)/np.sqrt(nodes) for shape in kernel_shapes]
        # self.kernels = [np.ones(shape) * 0.5 for shape in kernel_shapes]
        self.alpha = alpha
    
    def forward_propagations(self, xs): # For a big chunk of x, with same kernel, return an iterator
        if xs.ndim == 2:
            xs = xs[None, ...]
        self.last_input = xs
        return n_3d_convolutions(xs, self.kernels)

    def backward_propagation(self, dL_dys, dx=True):
        """
            We use the the partial derivatives of the
            cost function wrt to the weights to update
            the weights.
            And we return the derivatices of the cost
            function wrt to input.
        """
        weight_deltas = [np.zeros_like(k) for k in self.kernels]
        dL_dy_total = [0 for _ in self.kernels]

        for x, dL_dy in zip(self.last_input, dL_dys):
            dL_dw = n_convolutions(x, dL_dy)
            for i, d in enumerate(dL_dw):
                weight_deltas[i] += d
            
            if dx:
                dL_dy_total = [d + y for d, y in zip(dL_dy_total, dL_dy)]

        if dx: # We need to calculate this before the weight change
            dL_dx = sum([
                one_convolution(w, np.flip(y), padding=True) 
                for w, y in zip(self.kernels, dL_dy_total)
            ])
        else:
            dL_dx = 0

        self.kernels = [
            k - self.alpha * up
            for k, up 
            in zip(self.kernels, weight_deltas)
        ]
        return dL_dx
