#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rvv_simd.h"

namespace py = pybind11;

// Helper function to convert NumPy array to float pointer
float* numpy_to_float_ptr(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    return static_cast<float*>(buf.ptr);
}

// Helper function to get dimensions from NumPy array
std::vector<size_t> get_dimensions(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    std::vector<size_t> dims;
    for (auto d : buf.shape) {
        dims.push_back(static_cast<size_t>(d));
    }
    return dims;
}

// Helper function to create NumPy array from dimensions
py::array_t<float> create_numpy_array(const std::vector<size_t>& dims) {
    std::vector<py::ssize_t> py_dims;
    for (auto d : dims) {
        py_dims.push_back(static_cast<py::ssize_t>(d));
    }
    return py::array_t<float>(py_dims);
}

PYBIND11_MODULE(rvv_simd, m) {
    m.doc() = "Python bindings for RVV-SIMD: RISC-V Vector SIMD Library";
    
    // Core functions
    m.def("initialize", &rvv_simd::initialize, "Initialize the RVV-SIMD library");
    m.def("is_rvv_supported", &rvv_simd::is_rvv_supported, "Check if RVV is supported");
    m.def("get_version", &rvv_simd::get_version, "Get library version");
    m.def("get_rvv_info", &rvv_simd::get_rvv_info, "Get RVV implementation info");
    
    // Vector operations
    m.def("vector_add", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 1 || buf_b.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Input shapes must match");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_add(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Add two vectors element-wise");
    
    m.def("vector_sub", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 1 || buf_b.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Input shapes must match");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_sub(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Subtract two vectors element-wise");
    
    m.def("vector_mul", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 1 || buf_b.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Input shapes must match");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_mul(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Multiply two vectors element-wise");
    
    m.def("vector_div", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 1 || buf_b.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Input shapes must match");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_div(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Divide two vectors element-wise");
    
    m.def("vector_dot", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 1 || buf_b.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Input shapes must match");
        
        return rvv_simd::vector_dot(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0]
        );
    }, "Compute dot product of two vectors");
    
    m.def("vector_scale", [](py::array_t<float> a, float scalar) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_scale(
            static_cast<float*>(buf_a.ptr),
            scalar,
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Scale a vector by a scalar value");
    
    m.def("vector_norm", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        return rvv_simd::vector_norm(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0]
        );
    }, "Compute L2 norm of a vector");
    
    m.def("vector_normalize", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_normalize(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Normalize a vector to unit length");
    
    m.def("vector_exp", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_exp(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply exponential function to each element of a vector");
    
    m.def("vector_log", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_log(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply natural logarithm to each element of a vector");
    
    m.def("vector_sigmoid", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_sigmoid(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply sigmoid function to each element of a vector");
    
    m.def("vector_tanh", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_tanh(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply tanh function to each element of a vector");
    
    m.def("vector_relu", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        auto result = py::array_t<float>(buf_a.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::vector_relu(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply ReLU function to each element of a vector");
    
    // Matrix operations
    m.def("matrix_add", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 2 || buf_b.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        if (buf_a.shape[0] != buf_b.shape[0] || buf_a.shape[1] != buf_b.shape[1])
            throw std::runtime_error("Input shapes must match");
        
        auto result = py::array_t<float>({buf_a.shape[0], buf_a.shape[1]});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::matrix_add(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            buf_a.shape[1],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Add two matrices element-wise");
    
    m.def("matrix_sub", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 2 || buf_b.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        if (buf_a.shape[0] != buf_b.shape[0] || buf_a.shape[1] != buf_b.shape[1])
            throw std::runtime_error("Input shapes must match");
        
        auto result = py::array_t<float>({buf_a.shape[0], buf_a.shape[1]});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::matrix_sub(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            buf_a.shape[1],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Subtract two matrices element-wise");
    
    m.def("matrix_elem_mul", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 2 || buf_b.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        if (buf_a.shape[0] != buf_b.shape[0] || buf_a.shape[1] != buf_b.shape[1])
            throw std::runtime_error("Input shapes must match");
        
        auto result = py::array_t<float>({buf_a.shape[0], buf_a.shape[1]});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::matrix_elem_mul(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            buf_a.shape[1],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Multiply two matrices element-wise (Hadamard product)");
    
    m.def("matrix_mul", [](py::array_t<float> a, py::array_t<float> b) {
        py::buffer_info buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 2 || buf_b.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        if (buf_a.shape[1] != buf_b.shape[0])
            throw std::runtime_error("Inner dimensions must match for matrix multiplication");
        
        auto result = py::array_t<float>({buf_a.shape[0], buf_b.shape[1]});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::matrix_mul(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.shape[0],
            buf_a.shape[1],
            buf_b.shape[1],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Perform matrix multiplication");
    
    m.def("matrix_transpose", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        
        auto result = py::array_t<float>({buf_a.shape[1], buf_a.shape[0]});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::matrix_transpose(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            buf_a.shape[1],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Transpose a matrix");
    
    m.def("matrix_scale", [](py::array_t<float> a, float scalar) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        
        auto result = py::array_t<float>({buf_a.shape[0], buf_a.shape[1]});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::matrix_scale(
            static_cast<float*>(buf_a.ptr),
            scalar,
            buf_a.shape[0],
            buf_a.shape[1],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Scale a matrix by a scalar value");
    
    m.def("matrix_sum", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        
        return rvv_simd::matrix_sum(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            buf_a.shape[1]
        );
    }, "Compute the sum of all elements in a matrix");
    
    m.def("matrix_norm", [](py::array_t<float> a) {
        py::buffer_info buf_a = a.request();
        if (buf_a.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        
        return rvv_simd::matrix_norm(
            static_cast<float*>(buf_a.ptr),
            buf_a.shape[0],
            buf_a.shape[1]
        );
    }, "Compute the Frobenius norm of a matrix");
    
    // ML operations
    m.def("convolution_2d", [](py::array_t<float> input, py::array_t<float> kernel,
                              size_t stride_h, size_t stride_w,
                              size_t padding_h, size_t padding_w) {
        py::buffer_info buf_input = input.request(), buf_kernel = kernel.request();
        if (buf_input.ndim != 3)
            throw std::runtime_error("Input must have 3 dimensions (channels, height, width)");
        if (buf_kernel.ndim != 4)
            throw std::runtime_error("Kernel must have 4 dimensions (num_kernels, channels, height, width)");
        
        size_t input_c = buf_input.shape[0];
        size_t input_h = buf_input.shape[1];
        size_t input_w = buf_input.shape[2];
        
        size_t kernel_n = buf_kernel.shape[0];
        size_t kernel_c = buf_kernel.shape[1];
        size_t kernel_h = buf_kernel.shape[2];
        size_t kernel_w = buf_kernel.shape[3];
        
        if (input_c != kernel_c)
            throw std::runtime_error("Input and kernel channel dimensions must match");
        
        size_t output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
        size_t output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
        
        auto result = py::array_t<float>({kernel_n, output_h, output_w});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::convolution_2d(
            static_cast<float*>(buf_input.ptr),
            static_cast<float*>(buf_kernel.ptr),
            input_h, input_w, input_c,
            kernel_h, kernel_w, kernel_c, kernel_n,
            stride_h, stride_w,
            padding_h, padding_w,
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Perform 2D convolution operation");
    
    m.def("max_pooling_2d", [](py::array_t<float> input,
                              size_t pool_h, size_t pool_w,
                              size_t stride_h, size_t stride_w) {
        py::buffer_info buf_input = input.request();
        if (buf_input.ndim != 3)
            throw std::runtime_error("Input must have 3 dimensions (channels, height, width)");
        
        size_t input_c = buf_input.shape[0];
        size_t input_h = buf_input.shape[1];
        size_t input_w = buf_input.shape[2];
        
        size_t output_h = (input_h - pool_h) / stride_h + 1;
        size_t output_w = (input_w - pool_w) / stride_w + 1;
        
        auto result = py::array_t<float>({input_c, output_h, output_w});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::max_pooling_2d(
            static_cast<float*>(buf_input.ptr),
            input_h, input_w, input_c,
            pool_h, pool_w,
            stride_h, stride_w,
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Perform max pooling operation");
    
    m.def("avg_pooling_2d", [](py::array_t<float> input,
                              size_t pool_h, size_t pool_w,
                              size_t stride_h, size_t stride_w) {
        py::buffer_info buf_input = input.request();
        if (buf_input.ndim != 3)
            throw std::runtime_error("Input must have 3 dimensions (channels, height, width)");
        
        size_t input_c = buf_input.shape[0];
        size_t input_h = buf_input.shape[1];
        size_t input_w = buf_input.shape[2];
        
        size_t output_h = (input_h - pool_h) / stride_h + 1;
        size_t output_w = (input_w - pool_w) / stride_w + 1;
        
        auto result = py::array_t<float>({input_c, output_h, output_w});
        py::buffer_info buf_result = result.request();
        
        rvv_simd::avg_pooling_2d(
            static_cast<float*>(buf_input.ptr),
            input_h, input_w, input_c,
            pool_h, pool_w,
            stride_h, stride_w,
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Perform average pooling operation");
    
    m.def("batch_norm", [](py::array_t<float> input, py::array_t<float> gamma,
                          py::array_t<float> beta, py::array_t<float> mean,
                          py::array_t<float> var, float epsilon) {
        py::buffer_info buf_input = input.request();
        py::buffer_info buf_gamma = gamma.request();
        py::buffer_info buf_beta = beta.request();
        py::buffer_info buf_mean = mean.request();
        py::buffer_info buf_var = var.request();
        
        if (buf_input.ndim < 2)
            throw std::runtime_error("Input must have at least 2 dimensions");
        
        size_t channels = buf_gamma.shape[0];
        size_t size = buf_input.size / channels;
        
        if (buf_gamma.shape[0] != buf_beta.shape[0] ||
            buf_gamma.shape[0] != buf_mean.shape[0] ||
            buf_gamma.shape[0] != buf_var.shape[0])
            throw std::runtime_error("Parameter dimensions must match");
        
        auto result = py::array_t<float>(buf_input.shape);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::batch_norm(
            static_cast<float*>(buf_input.ptr),
            static_cast<float*>(buf_gamma.ptr),
            static_cast<float*>(buf_beta.ptr),
            static_cast<float*>(buf_mean.ptr),
            static_cast<float*>(buf_var.ptr),
            epsilon,
            size,
            channels,
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply batch normalization");
    
    m.def("softmax", [](py::array_t<float> input) {
        py::buffer_info buf_input = input.request();
        if (buf_input.ndim != 1)
            throw std::runtime_error("Input must be a 1D array");
        
        auto result = py::array_t<float>(buf_input.shape[0]);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::softmax(
            static_cast<float*>(buf_input.ptr),
            buf_input.shape[0],
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply softmax function to a vector");
    
    m.def("cross_entropy_loss", [](py::array_t<float> predictions, py::array_t<float> targets) {
        py::buffer_info buf_pred = predictions.request();
        py::buffer_info buf_targets = targets.request();
        
        if (buf_pred.ndim != 2 || buf_targets.ndim != 2)
            throw std::runtime_error("Inputs must be 2D arrays");
        
        if (buf_pred.shape[0] != buf_targets.shape[0] || buf_pred.shape[1] != buf_targets.shape[1])
            throw std::runtime_error("Input shapes must match");
        
        return rvv_simd::cross_entropy_loss(
            static_cast<float*>(buf_pred.ptr),
            static_cast<float*>(buf_targets.ptr),
            buf_pred.shape[0],
            buf_pred.shape[1]
        );
    }, "Compute cross-entropy loss");
    
    m.def("compute_gradients", [](py::array_t<float> output_grad, py::array_t<float> input) {
        py::buffer_info buf_grad = output_grad.request();
        py::buffer_info buf_input = input.request();
        
        if (buf_grad.ndim != buf_input.ndim)
            throw std::runtime_error("Input dimensions must match");
        
        for (size_t i = 0; i < buf_grad.ndim; i++) {
            if (buf_grad.shape[i] != buf_input.shape[i])
                throw std::runtime_error("Input shapes must match");
        }
        
        auto result = py::array_t<float>(buf_input.shape);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::compute_gradients(
            static_cast<float*>(buf_grad.ptr),
            static_cast<float*>(buf_input.ptr),
            buf_input.size,
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Compute gradients for backpropagation");
    
    m.def("apply_dropout", [](py::array_t<float> input, py::array_t<unsigned char> mask, float dropout_rate) {
        py::buffer_info buf_input = input.request();
        py::buffer_info buf_mask = mask.request();
        
        if (buf_input.size != buf_mask.size)
            throw std::runtime_error("Input and mask sizes must match");
        
        auto result = py::array_t<float>(buf_input.shape);
        py::buffer_info buf_result = result.request();
        
        rvv_simd::apply_dropout(
            static_cast<float*>(buf_input.ptr),
            static_cast<unsigned char*>(buf_mask.ptr),
            buf_input.size,
            dropout_rate,
            static_cast<float*>(buf_result.ptr)
        );
        
        return result;
    }, "Apply dropout during training");
}
