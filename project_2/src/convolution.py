import numpy as np
from scipy import signal, sparse
import time, itertools

class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.timings = {}
        self.counts = {}
    
    def time(self, part):
        self.timings[part] = self.timings.get(part, 0) + time.time() - self.start_time
        self.counts[part] = self.counts.get(part, 0) + 1
        self.start_time = time.time()
    
    def __str__(self):
        return self.timings.__str__() + "\n" + self.counts.__str__()

rc = Timer()

def shape_indexs(shape):
    return itertools.product(*[range(i) for i in shape])

def loop_convolution(input_image, kernel):
    """
    Convultion by looping over the indexes 
    and calculating the convolution 1 cell at the time.
    """
    rc.time("wait")
    out_put_image = np.zeros(
            (
                input_image.shape[0] - kernel.shape[0] + 1, #We cut of the edges
                input_image.shape[1] - kernel.shape[1] + 1
            )
        )

    for index in shape_indexs(out_put_image.shape):
        out_put_image[index] = (
            input_image[index[0]:index[0] + kernel.shape[0],
                        index[1]:index[1] + kernel.shape[1]] * kernel).sum()
    rc.time("loop")
    return out_put_image

def loop_convolution_2(input_image, kernel):
    rc.time("wait")
    out_put_image = np.zeros(
            (
                input_image.shape[0] - kernel.shape[0] + 1, #We cut of the edges
                input_image.shape[1] - kernel.shape[1] + 1
            )
        )

    for index in shape_indexs(kernel.shape):
        rc.time("out_put_nd")
        out_put_slice = input_image[
            index[0]:out_put_image.shape[0] + index[0],
            index[1]:out_put_image.shape[1] + index[1]
            ] * kernel[index]
        rc.time("out_put_slice")
        out_put_image += out_put_slice
        rc.time("out_put_add")
    return out_put_image

def scipy_convolution(input_image, kernel):
    """
    Convultion by using out of the box scipy function.
    This uses FFT for the convolution, if the matrix is big.
    Else direct calculation.
    """
    return signal.correlate(input_image, kernel, mode='valid')

def matrix_mul_convolution(input_image, kernel):
    """
    Convultion by creating a dense convolution matrix, this will use alot of memory.
    """
    kernel_mat = np.zeros(
            (
                np.prod([i - k + 1 for i, k in zip(input_image.shape, kernel.shape)]),
                input_image.shape[0] * input_image.shape[1]
            )
        )
    
    output_shape = (
        input_image.shape[0] - kernel.shape[0] + 1, #We cut of the edges
        input_image.shape[1] - kernel.shape[1] + 1
    )

    flattend_kernel = np.zeros(input_image.shape[1] * (kernel.shape[0] - 1) + kernel.shape[1])
    for row_i, kernel_row in enumerate(kernel):
        flattend_kernel[row_i * input_image.shape[1]
                        :row_i * input_image.shape[1] + kernel.shape[1]] = kernel_row

    for row, index in enumerate(shape_indexs(output_shape)):
        start_index = index[1] + index[0] * input_image.shape[1]
        kernel_mat[row, start_index: start_index+flattend_kernel.shape[0]] = flattend_kernel
    return (kernel_mat @ input_image.flatten()).reshape(output_shape)

def sparse_matrix_mul_convolution(input_image, kernel):
    """
    Convultion by creating a sparse convolution matrix.
    """
    input_size = input_image.shape[0] * input_image.shape[1]

    output_shape = (
        input_image.shape[0] - kernel.shape[0] + 1, #We cut of the edges
        input_image.shape[1] - kernel.shape[1] + 1
    )

    flattend_kernel = [k * np.ones(input_size) for k in kernel.flatten().tolist()]
    offsets = [ind[1] + ind[0] * input_image.shape[1] for ind in shape_indexs(kernel.shape)]
    kernel_sparse = sparse.dia_array(
        (flattend_kernel, offsets), 
        shape=(input_size, input_size)
        ).tocsr()

    row_selection = [index[1] + index[0] * input_image.shape[1] 
                     for index in shape_indexs(output_shape)]
    kernel_sparse = kernel_sparse[row_selection]

    return (kernel_sparse @ input_image.flatten()).reshape(output_shape)

def roll_matrix_convolution(input_image, kernel):
    """
    Some numpy rolling magic to first extend into 3D,
    Then for each item in the third dimension we shift,
    to align all values for the convolution in the top left corner.
    Then we sum out the new dimension.

    This is not memory efficient, but we can use the underlying
    matrix multiplication power of numpy. 
    """
    rc.time("wait")
    multiplied_images = input_image[..., None] * kernel.flatten() # We Extend the image into the 3rd dimension, with length of the flattend kernel
    rc.time("roll_mul")
    roll_numbers = [(-ind[0], -ind[1]) for ind in shape_indexs(kernel.shape)] # The shifting is the reverse of the index.
    # print(roll_numbers)
    rc.time("roll_num")
    for i, roll_n in enumerate(roll_numbers): # Shift each matrix along the 3rd dimension.
        multiplied_images[..., i] = np.roll(multiplied_images[...,i], shift=roll_n, axis=(0,1))
    rc.time("roll_roll")
    # Sum away the new dimension, and slice only the valid values (will be a wrapping convolution else)
    res = multiplied_images.sum(-1)[:input_image.shape[0] - kernel.shape[0] + 1,
                            :input_image.shape[1] - kernel.shape[1] + 1]
    rc.time("roll_res")
    return res

def one_convolution(input_image, kernel, method="loop", padding=False):
    """Handles the different kind of convolution methods"""
    if not padding:
        if input_image.shape[0] < kernel.shape[0] or input_image.shape[1] < kernel.shape[1]:
            raise IndexError("Kernel is bigger then the image.")
    else:
        padding_width = [(s - 1, s - 1) for s in kernel.shape]
        input_image = np.pad(input_image, padding_width)
    if method == "scipy":
        return scipy_convolution(input_image, kernel)
    elif method == "loop_alt":
        return loop_convolution_2(input_image, kernel)
    elif method == "loop":
        return loop_convolution(input_image, kernel)
    elif method == "sparse_mul":
        return sparse_matrix_mul_convolution(input_image, kernel)
    elif method == "roll":
        return roll_matrix_convolution(input_image, kernel)

def loop_n_convolutions(input_image, kernels):
    rc.time("wait")
    out_put_image = np.zeros((
            kernels.shape[0],
            input_image.shape[0] - kernels.shape[1] + 1, #We cut of the edges
            input_image.shape[1] - kernels.shape[2] + 1
    ))

    for index in shape_indexs(out_put_image.shape[1:]):
        index_slice = input_image[index[0]:index[0] + kernels.shape[1],
                        index[1]:index[1] + kernels.shape[2]]
        out_put_image[(slice(None),) + index] = (index_slice[None, ...] * kernels).sum((1, 2))
    
    rc.time("loop_n")
    return out_put_image

def loop_n_convolutions_2(input_image, kernels):
    rc.time("wait")
    out_put_image = np.zeros((
            kernels.shape[0],
            input_image.shape[0] - kernels.shape[1] + 1, #We cut of the edges
            input_image.shape[1] - kernels.shape[2] + 1
    ))
    for index in shape_indexs(kernels.shape[1:]):
        out_put_slice = input_image[
            index[0]:out_put_image.shape[1] + index[0],
            index[1]:out_put_image.shape[2] + index[1]
            ] * kernels[(slice(None),) + index + (None, None)]
        out_put_image += out_put_slice
    rc.time("loop_n")
    return out_put_image

def n_convolutions(input_image, kernels, method="loop", padding=False):
    if not padding:
        if input_image.shape[0] < kernels[0].shape[0] or input_image.shape[1] < kernels[0].shape[1]:
            raise IndexError("Kernel is bigger then the image.")
    else:
        padding_width = [(s - 1, s - 1) for s in kernels[0].shape]
        input_image = np.pad(input_image, padding_width)

    return loop_n_convolutions(input_image, kernels)

def n_3d_convolutions(input_images, kernels):
    """
    Convultion by creating a sparse convolution matrices,
    then applying for each image. Returns as an iterator.
    """
    rc.time("wait")
    k_matrices = []
    output_shape = (
        input_images[0].shape[0] - kernels.shape[1] + 1, #We cut of the edges
        input_images[0].shape[1] - kernels.shape[2] + 1
    )
    for kernel in kernels:
        input_size = input_images[0].shape[0] * input_images[0].shape[1]

        flattend_kernel = [k * np.ones(input_size) for k in kernel.flatten().tolist()]
        offsets = [ind[1] + ind[0] * input_images[0].shape[1] for ind in shape_indexs(kernel.shape)]
        kernel_sparse = sparse.dia_array(
            (flattend_kernel, offsets), 
            shape=(input_size, input_size)
            ).tocsr()

        row_selection = [index[1] + index[0] * input_images[0].shape[1] 
                        for index in shape_indexs(output_shape)]
        kernel_sparse = kernel_sparse[row_selection]
        k_matrices.append(kernel_sparse)

    k_matrix = sparse.vstack(k_matrices)
    rc.time("matrix_setup")
    for x in input_images:
        rc.time("wait")
        res = k_matrix @ x.flatten()
        rc.time("matrix_mul")
        yield res.reshape((-1,) + output_shape)


def window_max(x, window_shape, strides):
    rc.time("wait")
    out_put = np.zeros((
        x.shape[0],
        (x.shape[1] - window_shape[0])//strides[0] + 1,
        (x.shape[2] - window_shape[1])//strides[1] + 1
        ))
    max_idxs = []

    for index in shape_indexs(out_put.shape[1:]):
        out_put_slice = x[
            :,
            strides[0] * index[0]: strides[0] * index[0] + window_shape[0],
            strides[1] * index[1]: strides[1] * index[1] + window_shape[1],
            ]
        rc.time("window_slice")
        for i, s in enumerate(out_put_slice):
            ind = np.unravel_index(np.argmax(s, axis=None), s.shape)
            max_idxs.append((
                (i,) + index,
                (i, index[0] * strides[0] + ind[0], index[1] * strides[1] + ind[1])
            ))
            rc.time("max_idx")
            out_put[(i,) + index] = out_put_slice[(i,) + ind]
            rc.time("window_max")
    return out_put, max_idxs

if __name__ == "__main__":
    # np.random.seed(1)
    random_image = np.floor(np.random.rand(3,4,4)*10)
    # random_image = np.arange(32, 0, -1).reshape((4,4,2))
    print(random_image)
    print(window_max(random_image, (2,2), (2,2))[0])
    # random_images = [np.random.rand(28, 28) for _ in range(100)]
    # test_kernels = np.stack([np.array([[0,0,0],[0,i,0],[0,0,0]]) for i in range(1,3)])

    # for im in random_images:
    #     a = loop_n_convolutions_2(im, test_kernels)
    #     b = loop_n_convolutions(im, test_kernels)
    #     print([(x - y).sum() for x, y in zip(a,b)])