import numpy as np
from scipy import signal, sparse
import time, itertools

def shape_indexs(shape):
    return itertools.product(*[range(i) for i in shape])

def loop_convolution(input_image, kernel):
    """
    Convultion by looping over the indexes and calculating the convolution 1 cell at the time.
    """
    out_put_image = np.zeros(
            (
                input_image.shape[0] - kernel.shape[0] + 1, #We cut of the edges
                input_image.shape[1] - kernel.shape[1] + 1
            )
        )

    for index in shape_indexs(out_put_image.shape):
        input_slice = input_image[index[0]:index[0] + kernel.shape[0], index[1]:index[1] + kernel.shape[1]]
        out_put_image[index] = (input_slice * kernel).sum()

    return out_put_image

def scipy_convolution(input_image, kernel):
    """
    Convultion by using out of the box scipy function.
    This uses FFT for the convolution, if the matrix is big.
    Else direct calculation.
    """
    return signal.convolve(input_image, np.flip(kernel), mode='valid')

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
        flattend_kernel[row_i * input_image.shape[1]: row_i * input_image.shape[1] + kernel.shape[1]] = kernel_row

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
    kernel_sparse = sparse.dia_array((flattend_kernel, offsets), shape=(input_size, input_size)).tocsr()

    row_selection = [index[1] + index[0] * input_image.shape[1] for index in shape_indexs(output_shape)]
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
    multiplied_images = input_image[..., None] * kernel.flatten() # We Extend the image into the 3rd dimension, with length of the flattend kernel
    roll_numbers = [(-ind[0], -ind[1]) for ind in shape_indexs(kernel.shape)] # The shifting is the reverse of the index.
    for i, roll_n in enumerate(roll_numbers): # Shift each matrix along the 3rd dimension.
        multiplied_images[..., i] = np.roll(multiplied_images[...,i], shift=roll_n, axis=(0,1))

    # Slice the to the valid values, and sum away the new dimension
    res = multiplied_images[:input_image.shape[0] - kernel.shape[0] + 1, :input_image.shape[1] - kernel.shape[1] + 1].sum(-1)
    return res

def convolution(input_image, kernel, method="scipy"):
    if input_image.shape[0] < kernel.shape[0] or input_image.shape[1] < kernel.shape[1]:
        raise IndexError("Kernel is bigger then the image.")
    
    """Handles the different kind of convolution methods"""
    if method == "scipy":
        return scipy_convolution(input_image, kernel)
    elif method == "sparse_mul":
        return sparse_matrix_mul_convolution(input_image, kernel)
    elif method == "loop":
        return loop_convolution(input_image, kernel)
    
def n_convolutions(input_image, kernels, method="scipy"):
    """
        Does n convolutions with each kernel.
        TODO: Do not do sequential.
    """
    return [convolution(input_image, kernel, method=method) for kernel in kernels]

if __name__ == "__main__":
    # test_image = np.arange(25).reshape((5,5))
    # test_image = np.arange(25).reshape((5,5))
    # test_kernel = np.array([[0,2,0],[-1,4,-2],[3,0,1]])
    # print(scipy_convolution(test_image, test_kernel))
    # print(roll_matrix_convolution(test_image, test_kernel))
    # print(loop_convolution(test_image, test_kernel))
    # print(sparse_matrix_mul_convolution(test_image, test_kernel))

    np.random.seed(1)
    random_image = np.random.rand(1000, 10000)
    test_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    t1 = time.time()
    convolution1 = scipy_convolution(random_image, test_kernel)
    print("scipy took:", time.time() - t1)
    t1 = time.time()
    convolution2 = loop_convolution(random_image, test_kernel)
    print("loop took:", time.time() - t1)
    t1 = time.time()
    # convolution3 = matrix_mul_convolution(random_image, test_kernel)
    convolution4 = sparse_matrix_mul_convolution(random_image, test_kernel)
    print("sparse took:", time.time() - t1)
    t1 = time.time()
    convolution5 = roll_matrix_convolution(random_image, test_kernel)
    print("rotate took:", time.time() - t1)