import numpy as np
from scipy import signal, sparse
import time, itertools

from functools import wraps

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
    # print(method)
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
    kernels = np.stack(kernels, axis=-1)
    out_put_image = np.zeros((
            input_image.shape[0] - kernels.shape[0] + 1, #We cut of the edges
            input_image.shape[1] - kernels.shape[1] + 1,
            kernels.shape[2]
    ))
    for index in shape_indexs(out_put_image.shape[:-1]):
        index_slice = input_image[index[0]:index[0] + kernels.shape[0],
                        index[1]:index[1] + kernels.shape[1]]
        out_put_image[index] = (index_slice[..., None] * kernels).sum((0, 1))
    rc.time("loop_n")
    return list(map(np.squeeze, np.split(out_put_image, out_put_image.shape[-1], -1)))

def n_convolutions(input_image, kernels, method="loop", padding=False, old=True):
    """
        Does n convolutions with each kernel.
        TODO: Do not do sequential.
    """
    if old:
        return [one_convolution(input_image, kernel, method=method, padding=padding) for kernel in kernels]

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
    out_shapes = []
    for kernel in kernels:
        input_size = input_images[0].shape[0] * input_images[0].shape[1]

        output_shape = (
            input_images[0].shape[0] - kernel.shape[0] + 1, #We cut of the edges
            input_images[0].shape[1] - kernel.shape[1] + 1
        )

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
        out_shapes.append(output_shape)

    slice_shapes = []
    start = 0
    for s in out_shapes:
        slice_shapes.append((slice(start, start + np.prod(s)), s))
        start += np.prod(s)

    k_matrix = sparse.vstack(k_matrices)
    rc.time("matrix_setup")
    for x in input_images:
        rc.time("wait")
        res = k_matrix @ x.flatten()
        rc.time("matrix_mul")
        yield [res[slc].reshape(shape) for slc, shape in slice_shapes]

if __name__ == "__main__":
    np.random.seed(1)
    random_image = np.random.rand(300, 300)
    # random_image = np.arange(16).reshape((4,4))
    test_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    a = loop_convolution(random_image, test_kernel)
    b = loop_convolution_2(random_image, test_kernel)
    print((a- b).sum())
    # test_kernel = np.arange(9).reshape((3,3))
    # print((ind- rll).sum())
    print(rc)
    # random_images = [np.random.rand(28, 28) for _ in range(10000)]
    # test_kernels = [np.array([[0,0,0],[0,i,0],[0,0,0]]) for i in range(5)]
    # t1 = time.time()
    # r = loop_n_convolutions(random_image, test_kernels)
    # print("new", time.time() - t1)
    # t1 = time.time()
    # r2 = n_convolutions(random_image, test_kernels)
    # print("old", time.time() - t1)

    # t1 = time.time()
    # for im in random_images:
    #     convolution1 = scipy_convolution(im, test_kernel)
    # print("scipy took:", (time.time() - t1) * 1000)
    # t1 = time.time()
    # for im in random_images:
    #     convolution2 = loop_convolution(im, test_kernel)
    # print("loop took:", (time.time() - t1) * 1000)
    # t1 = time.time()
    # for im in random_images:
    #     convolution4 = sparse_matrix_mul_convolution(im, test_kernel)
    # print("sparse took:", (time.time() - t1) * 1000)
    # t1 = time.time()
    # for im in random_images:
    #     convolution5 = roll_matrix_convolution(im, test_kernel)
    # print("rotate took:", (time.time() - t1) * 1000)