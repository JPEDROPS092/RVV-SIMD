"""
RVV-SIMD: RISC-V Vector SIMD Library Python Bindings

This package provides Python bindings for the RVV-SIMD library,
enabling efficient vector and matrix operations using RISC-V Vector extensions.
"""

from .rvv_simd import (
    # Core functions
    initialize, is_rvv_supported, get_version, get_rvv_info,
    
    # Vector operations
    vector_add, vector_sub, vector_mul, vector_div,
    vector_dot, vector_scale, vector_norm, vector_normalize,
    vector_exp, vector_log, vector_sigmoid, vector_tanh, vector_relu,
    
    # Matrix operations
    matrix_add, matrix_sub, matrix_elem_mul, matrix_mul,
    matrix_transpose, matrix_scale, matrix_sum, matrix_norm,
    
    # ML operations
    convolution_2d, max_pooling_2d, avg_pooling_2d,
    batch_norm, softmax, cross_entropy_loss,
    compute_gradients, apply_dropout
)

# Initialize the library when imported
initialize()

# NumPy-like convenience functions
def add(a, b):
    """Element-wise addition of arrays."""
    return vector_add(a, b)

def subtract(a, b):
    """Element-wise subtraction of arrays."""
    return vector_sub(a, b)

def multiply(a, b):
    """Element-wise multiplication of arrays."""
    return vector_mul(a, b)

def divide(a, b):
    """Element-wise division of arrays."""
    return vector_div(a, b)

def dot(a, b):
    """Dot product of vectors."""
    return vector_dot(a, b)

def matmul(a, b):
    """Matrix multiplication."""
    return matrix_mul(a, b)

def transpose(a):
    """Matrix transpose."""
    return matrix_transpose(a)

def relu(x):
    """ReLU activation function."""
    return vector_relu(x)

def sigmoid(x):
    """Sigmoid activation function."""
    return vector_sigmoid(x)

def tanh(x):
    """Tanh activation function."""
    return vector_tanh(x)

def exp(x):
    """Exponential function."""
    return vector_exp(x)

def log(x):
    """Natural logarithm function."""
    return vector_log(x)

def norm(x):
    """Compute the norm of a vector or matrix."""
    return vector_norm(x) if len(x.shape) == 1 else matrix_norm(x)

def conv2d(input, kernel, stride=(1, 1), padding=(0, 0)):
    """2D convolution operation."""
    return convolution_2d(input, kernel, stride[0], stride[1], padding[0], padding[1])

def max_pool2d(input, kernel_size, stride=None):
    """2D max pooling operation."""
    if stride is None:
        stride = kernel_size
    return max_pooling_2d(input, kernel_size[0], kernel_size[1], stride[0], stride[1])

def avg_pool2d(input, kernel_size, stride=None):
    """2D average pooling operation."""
    if stride is None:
        stride = kernel_size
    return avg_pooling_2d(input, kernel_size[0], kernel_size[1], stride[0], stride[1])
