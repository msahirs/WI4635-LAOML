import numpy as np
from itertools import pairwise
from .convolution import n_convolutions, one_convolution, n_3d_convolutions



class ConvLay:
    def __init__(self, kernel_shapes, alpha) -> None:
        nodes = sum((np.prod(s) for s in kernel_shapes))
        self.kernels = [(np.random.rand(*shape) - 1/2)/np.sqrt(nodes) for shape in kernel_shapes]
        self.alpha = alpha

    def set_next_layer(self, layer):
        self.next = layer
        layer.set_previous_layer(self)

    def set_previous_layer(self, layer):
        self.previous = layer
    
    def forward_propagations(self, xs): # For a big chunk of x, with same kernel, return an iterator
        self.last_input = xs
        res = n_3d_convolutions(xs, self.kernels)
        # print("Conv forward")
        if hasattr(self, "next"):
            return self.next.forward_propagations(res)
        else:
            return res

    def backward_propagations(self, dL_dys, dx=False):
        """
            We use the the partial derivatives of the
            cost function wrt to the weights to update
            the weights.
            And we return the derivatices of the cost
            function wrt to input.
        """
        # print("Conv back")
        weight_deltas = [np.zeros_like(k) for k in self.kernels]
        dL_dy_total = [0 for _ in self.kernels]
        for x, dL_dy in zip(self.last_input, dL_dys):
            dL_dw = n_convolutions(x, dL_dy, old=True)
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

class MaxPool:
    def __init__(self, pool_shape=(2,2), stride=(1,1)):
        pass

    def set_next_layer(self, layer):
        self.next = layer
        layer.set_previous_layer(self)

    def set_previous_layer(self, layer):
        self.previous = layer

    def forward_propagations(self, xs):
        print("MinMax forward")
        self.last_input = xs
        res = xs
        if hasattr(self, "next"):
            return self.next.forward_propagations(res)
        else:
            return res
    
    def backward_propagations(self, dL_dy):
        print("MinMax back")
        res = dL_dy
        if hasattr(self, "previous"):
            return self.previous.backward_propagations(res)
        else:
            return res

class SoftMax:
    def __init__(self) -> None:
        pass

    def set_next_layer(self, layer):
        self.next = layer
        layer.set_previous_layer(self)

    def set_previous_layer(self, layer):
        self.previous = layer

    def forward_propagations(self, xs):
        print("SoftMax forward")
        self.last_input = xs
        res = xs
        if hasattr(self, "next"):
            return self.next.forward_propagations(res)
        else:
            return res
    
    def backward_propagations(self, dL_dy):
        print("SoftMax backwards")
        res = dL_dy
        if hasattr(self, "previous"):
            return self.previous.backward_propagations(res)
        else:
            return res
        
class ConvNN:
    lay_map = {
        "convolution": ConvLay,
        "min_max_pool":MaxPool,
        "soft_max":SoftMax
    }
    def __init__(self, config):
        self.layers = []
        for k, v in config.items():
            self.layers.append(self.lay_map[k](**v))
        
        for prev, next in pairwise(self.layers):
            prev.set_next_layer(next)
    
    def forward_propagations(self, xs):
        return self.layers[0].forward_propagations(xs)
    
    def backward_propagations(self, dL_dy):
        return self.layers[-1].backward_propagations(dL_dy)