import numpy as np
from itertools import pairwise
from .convolution import n_convolutions, one_convolution, n_3d_convolutions, window_max, Timer, window_avg, window_max_test

class ConvLay:
    def __init__(self, kernel_shapes, alpha, input_shape) -> None:
        nodes = sum((np.prod(s) for s in kernel_shapes))
        self.kernels = np.stack([(np.random.rand(*shape) - 1/2)/np.sqrt(nodes) for shape in kernel_shapes])
        self.alpha = alpha
        self.input_shape = input_shape

    @property
    def output_shape(self):
        return (
            self.kernels.shape[0],
            self.input_shape[0] - self.kernels.shape[1] + 1,
            self.input_shape[1] - self.kernels.shape[2] + 1
    )

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
        weight_deltas = np.zeros_like(self.kernels)
        dL_dy_total = [0 for _ in self.kernels]
        for x, dL_dy in zip(self.last_input, dL_dys):
            dL_dw = n_convolutions(x, dL_dy)
            weight_deltas += dL_dw
            
            if dx:
                dL_dy_total = [d + y for d, y in zip(dL_dy_total, dL_dy)]

        if dx: # We need to calculate this before the weight change
            dL_dx = sum([
                one_convolution(w, np.flip(y), padding=True) 
                for w, y in zip(self.kernels, dL_dy_total)
            ])
        else:
            dL_dx = 0
        self.kernels = self.kernels - self.alpha * weight_deltas
        if np.abs(self.kernels).max() > 1e6:
            raise RuntimeError("Diverging kernel")
        
        if hasattr(self, "previous"):
            return self.previous.backward_propagations(dL_dx)
        else:
            return dL_dx
        
    def save_obj(self):
        return {
            "kernels": self.kernels,
            "alpha": self.alpha,
        }

class MaxPool:
    def __init__(self, pool_shape, strides, input_shape):
        self.pool_shape = pool_shape
        self.strides = strides
        self.input_shape = input_shape

    @property
    def output_shape(self):
        return (
            self.input_shape[0],
            (self.input_shape[1] - self.pool_shape[0])//self.strides[0] + 1,
            (self.input_shape[2] - self.pool_shape[1])//self.strides[1] + 1,
        )

    def set_next_layer(self, layer):
        self.next = layer
        layer.set_previous_layer(self)

    def set_previous_layer(self, layer):
        self.previous = layer

    def window_iterator(self, xs):
        self.last_maxs = []
        self.last_shapes = []
        for x in xs:
            res, max_msk = window_max(x, self.pool_shape, self.strides)
            # res, max_msk = window_max_test(x, self.pool_shape, self.strides)
            self.last_maxs.append(max_msk)
            self.last_shapes.append(x.shape)
            yield res

    def forward_propagations(self, xs):
        self.last_input = xs
        res = self.window_iterator(xs)
        if hasattr(self, "next"):
            return self.next.forward_propagations(res)
        else:
            return res
        
    def back_iterator(self, dL_dys):
        for dL in dL_dys:
            max_indices = self.last_maxs.pop(0)
            res = np.zeros(self.last_shapes.pop(0))
            for src, tar in max_indices:
                res[tar] += dL[src]
            yield res

    def back_iterator_test(self, dL_dys):
        for dL in dL_dys:
            res = np.zeros(self.last_shapes.pop(0))
            max_indices = self.last_maxs.pop(0)
            int_c = 0
            # print(c, "of", len(dL_dys))
            for m in range(res.shape[0]):
                for h in range(dL.shape[0]):                   # loop on the vertical axis
                    for w in range(dL.shape[1]):               # loop on the horizontal axis
                        vert_start  = h*self.strides[0]
                        vert_end    = vert_start + self.pool_shape[0]
                        horiz_start = w*self.strides[1]
                        horiz_end   = horiz_start + self.pool_shape[1]

                        res[m,vert_start:vert_end, horiz_start:horiz_end] += np.multiply(max_indices[int_c], dL[m, h, w])
                        int_c += 1
                        # print(int_c)
            yield res
    
    def backward_propagations(self, dL_dys):
        res = self.back_iterator(dL_dys)
        # res = self.back_iterator_test(dL_dys)
        if hasattr(self, "previous"):
            return self.previous.backward_propagations(res)
        else:
            return res
        
    def save_obj(self):
        return {
            "pool_shape": self.pool_shape,
            "strides": self.strides,
        }
    
    
class AvgPool:
    def __init__(self, pool_shape, strides, input_shape):
        self.pool_shape = pool_shape
        self.strides = strides
        self.input_shape = input_shape

    @property
    def output_shape(self):
        return (
            self.input_shape[0],
            (self.input_shape[1] - self.pool_shape[0])//self.strides[0] + 1,
            (self.input_shape[2] - self.pool_shape[1])//self.strides[1] + 1,
        )

    def set_next_layer(self, layer):
        self.next = layer
        layer.set_previous_layer(self)

    def set_previous_layer(self, layer):
        self.previous = layer

    def window_iterator(self, xs):
        self.last_maxs = []
        self.last_shapes = []
        for x in xs:
            res = window_avg(x, self.pool_shape, self.strides)
            self.last_shapes.append(x.shape)
            yield res

    def forward_propagations(self, xs):
        self.last_input = xs
        res = self.window_iterator(xs)
        if hasattr(self, "next"):
            return self.next.forward_propagations(res)
        else:
            return res
    
    def distribute_avg(self, dL):
        
        avg = dL / (self.pool_shape[0] * self.pool_shape[1])

        return np.ones(self.pool_shape) * avg

    def back_iterator(self, dL_dys):
        for dL in dL_dys:
            res = np.zeros(self.last_shapes.pop(0))

            for m in range(res.shape[0]):
                for h in range(dL.shape[0]):                   # loop on the vertical axis
                    for w in range(dL.shape[1]):               # loop on the horizontal axis
                        vert_start  = h*self.strides[0]
                        vert_end    = vert_start + self.pool_shape[0]
                        horiz_start = w*self.strides[1]
                        horiz_end   = horiz_start + self.pool_shape[1]

                        res[m, vert_start:vert_end, horiz_start:horiz_end] += self.distribute_avg(dL[m, h, w])
            
            yield res
    
    def backward_propagations(self, dL_dys):
        res = self.back_iterator(dL_dys)
        if hasattr(self, "previous"):
            return self.previous.backward_propagations(res)
        else:
            return res
        
    def save_obj(self):
        return {
            "pool_shape": self.pool_shape,
            "strides": self.strides,
        }
    

class SoftMax:
    def __init__(self, input_shape, output_length, alpha) -> None:
        self.input_shape = input_shape
        self.output_shape = output_length
        self.alpha = alpha
        self.weights = np.zeros((output_length, np.prod(input_shape)))#np.random.rand(*)
        self.biases = np.zeros(output_length)
        self.last = False

    def set_next_layer(self, layer):
        self.next = layer
        layer.set_previous_layer(self)

    def set_previous_layer(self, layer):
        self.previous = layer

    def forward_iterator(self, xs):
        self.last_inputs = []
        self.last_res = []
        for x in xs:
            self.last_inputs.append(x)
            inp = np.exp(self.weights @ x.flatten() + self.biases)
            res = inp/inp.sum()
            if not self.last:
                self.last_res.append(res)
            yield res

    def forward_propagations(self, xs):
        res = self.forward_iterator(xs)
        if hasattr(self, "next"):
            return self.next.forward_propagations(res)
        else:
            return res
        
    def backward_last(self, ys):
        dL_dw = np.zeros_like(self.weights)
        dL_db = np.zeros_like(self.biases)
        dL_dxs = []
        for y_bar, y in ys:
            inp = self.last_inputs.pop(0)
            A = y_bar - y
            dL_dw += np.outer(A, inp.flatten())
            dL_db += A
            dL_dxs.append((A @ self.weights).reshape(self.input_shape))

        self.weights = self.weights - self.alpha * dL_dw
        self.biases = self.biases - self.alpha * dL_db
        if np.abs(self.weights).max() > 1e6 or np.abs(self.biases).max() > 1e6:
            raise RuntimeError("Diverging softmax")
        if hasattr(self, "previous"):
            return self.previous.backward_propagations(dL_dxs)
        else:
            return dL_dxs

    def backward_middle(self, dL_dys):
        dL_dw = np.zeros_like(self.weights)
        dL_db = np.zeros_like(self.biases)
        dL_dxs = []
        for dL_dy in dL_dys:
            inp = self.last_inputs.pop(0)
            res = self.last_res.pop(0)
            A = (np.outer(-res, res) + np.diag(res)) @ dL_dy
            dL_dw += np.outer(A, inp.flatten())
            dL_db += A
            dL_dxs.append((A @ self.weights).reshape(self.input_shape))

        self.weights = self.weights - self.alpha * dL_dw
        self.biases = self.biases - self.alpha * dL_db

        if hasattr(self, "previous"):
            return self.previous.backward_propagations(dL_dxs)
        else:
            return dL_dxs
        
    def backward_propagations(self, ys):
        if self.last:
            return self.backward_last(ys)
        else:
            return self.backward_middle(ys)
        
    def save_obj(self):
        return {
            "weights": self.weights,
            "biases": self.biases
        }
    
        
class ConvNN:
    lay_map = {
        "convolution": ConvLay,
        "min_max_pool":MaxPool,
        "avg_pool":AvgPool,
        "soft_max":SoftMax
    }

    def __init__(self, config, input_shape):
        self.layers = []
        self.lay_names = []
        for k, v in config.items():
            v.update({"input_shape":input_shape})
            self.layers.append(self.lay_map[k](**v))
            self.lay_names.append(k)
            input_shape = self.layers[-1].output_shape
        
        for prev, next in pairwise(self.layers):
            prev.set_next_layer(next)

        self.layers[-1].last = True
    
    def forward_propagations(self, xs):
        return self.layers[0].forward_propagations(xs)
    
    def backward_propagations(self, dL_dy):
        return self.layers[-1].backward_propagations(dL_dy)
    
    def save(self):
        objs = [(name, lay.save_obj) for name, lay in zip(self.lay_name, self.layers)]
        #write objs
    
    def load(self):
        pass