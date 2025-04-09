# RVV-SIMD Library Usage Guide

This guide provides instructions on how to use the RVV-SIMD library for RISC-V Vector operations in both C++ and Python applications.

## Table of Contents

1. [Installation](#installation)
2. [C++ Usage](#c-usage)
3. [Python Usage](#python-usage)
4. [Performance Optimization](#performance-optimization)
5. [Benchmarking](#benchmarking)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- RISC-V toolchain with vector extension support
- CMake (3.10+)
- Python 3.6+ (for Python bindings)
- pybind11 (for Python bindings)
- NumPy (for Python examples)

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rvv-simd.git
   cd rvv-simd
   ```

2. Create a build directory and build the library:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. Install the library:
   ```bash
   make install
   ```

### Installing Python Bindings

```bash
cd python
pip install -e .
```

## C++ Usage

### Including the Library

```cpp
#include "rvv_simd.h"
```

### Initializing the Library

```cpp
// Initialize the library
if (!rvv_simd::initialize()) {
    std::cerr << "Failed to initialize RVV-SIMD library" << std::endl;
    return 1;
}

// Check if RVV is supported
if (rvv_simd::is_rvv_supported()) {
    std::cout << "RVV is supported on this hardware" << std::endl;
} else {
    std::cout << "RVV is not supported, using fallback implementations" << std::endl;
}

// Get library information
std::cout << "RVV-SIMD Version: " << rvv_simd::get_version() << std::endl;
std::cout << "RVV Info: " << rvv_simd::get_rvv_info() << std::endl;
```

### Vector Operations

```cpp
// Create input vectors
const size_t size = 1024;
float* a = new float[size];
float* b = new float[size];
float* result = new float[size];

// Initialize vectors with data
// ...

// Vector addition
rvv_simd::vector_add(a, b, size, result);

// Vector dot product
float dot_product = rvv_simd::vector_dot(a, b, size);

// Vector scaling
rvv_simd::vector_scale(a, 2.5f, size, result);

// Vector normalization
rvv_simd::vector_normalize(a, size, result);

// Vector operations with activation functions
rvv_simd::vector_sigmoid(a, size, result);
rvv_simd::vector_relu(a, size, result);

// Clean up
delete[] a;
delete[] b;
delete[] result;
```

### Matrix Operations

```cpp
// Create input matrices
const size_t rows = 32;
const size_t cols = 32;
float* a = new float[rows * cols];
float* b = new float[rows * cols];
float* result = new float[rows * cols];

// Initialize matrices with data
// ...

// Matrix addition
rvv_simd::matrix_add(a, b, rows, cols, result);

// Matrix multiplication
const size_t a_rows = 32;
const size_t a_cols = 64;
const size_t b_cols = 32;
float* a_mat = new float[a_rows * a_cols];
float* b_mat = new float[a_cols * b_cols];
float* c_mat = new float[a_rows * b_cols];

rvv_simd::matrix_mul(a_mat, b_mat, a_rows, a_cols, b_cols, c_mat);

// Matrix transpose
float* a_transpose = new float[cols * rows];
rvv_simd::matrix_transpose(a, rows, cols, a_transpose);

// Clean up
delete[] a;
delete[] b;
delete[] result;
delete[] a_mat;
delete[] b_mat;
delete[] c_mat;
delete[] a_transpose;
```

### Machine Learning Operations

```cpp
// Convolution operation
const size_t input_c = 3;
const size_t input_h = 32;
const size_t input_w = 32;
const size_t kernel_n = 16;
const size_t kernel_h = 3;
const size_t kernel_w = 3;
const size_t stride_h = 1;
const size_t stride_w = 1;
const size_t padding_h = 1;
const size_t padding_w = 1;

float* input = new float[input_c * input_h * input_w];
float* kernel = new float[kernel_n * input_c * kernel_h * kernel_w];

// Calculate output dimensions
const size_t output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
const size_t output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
float* output = new float[kernel_n * output_h * output_w];

rvv_simd::convolution_2d(
    input, kernel,
    input_h, input_w, input_c,
    kernel_h, kernel_w, input_c, kernel_n,
    stride_h, stride_w,
    padding_h, padding_w,
    output
);

// Max pooling
const size_t pool_h = 2;
const size_t pool_w = 2;
const size_t pool_stride_h = 2;
const size_t pool_stride_w = 2;

// Calculate pooled output dimensions
const size_t pooled_h = (output_h - pool_h) / pool_stride_h + 1;
const size_t pooled_w = (output_w - pool_w) / pool_stride_w + 1;
float* pooled = new float[kernel_n * pooled_h * pooled_w];

rvv_simd::max_pooling_2d(
    output,
    output_h, output_w, kernel_n,
    pool_h, pool_w,
    pool_stride_h, pool_stride_w,
    pooled
);

// Clean up
delete[] input;
delete[] kernel;
delete[] output;
delete[] pooled;
```

## Python Usage

### Importing the Library

```python
import numpy as np
import rvv_simd as rv
```

### Vector Operations

```python
# Create input vectors
size = 1024
a = np.random.uniform(-10, 10, size).astype(np.float32)
b = np.random.uniform(-10, 10, size).astype(np.float32)

# Vector addition
c = rv.vector_add(a, b)
# or using the NumPy-like API
c = rv.add(a, b)

# Vector dot product
dot_product = rv.vector_dot(a, b)
# or using the NumPy-like API
dot_product = rv.dot(a, b)

# Vector scaling
scaled = rv.vector_scale(a, 2.5)

# Vector normalization
normalized = rv.vector_normalize(a)

# Vector operations with activation functions
sigmoid_result = rv.sigmoid(a)
relu_result = rv.relu(a)
```

### Matrix Operations

```python
# Create input matrices
rows, cols = 32, 32
a = np.random.uniform(-10, 10, (rows, cols)).astype(np.float32)
b = np.random.uniform(-10, 10, (rows, cols)).astype(np.float32)

# Matrix addition
c = rv.matrix_add(a, b)

# Matrix multiplication
a_rows, a_cols, b_cols = 32, 64, 32
a_mat = np.random.uniform(-10, 10, (a_rows, a_cols)).astype(np.float32)
b_mat = np.random.uniform(-10, 10, (a_cols, b_cols)).astype(np.float32)

c_mat = rv.matrix_mul(a_mat, b_mat)
# or using the NumPy-like API
c_mat = rv.matmul(a_mat, b_mat)

# Matrix transpose
a_transpose = rv.matrix_transpose(a)
# or using the NumPy-like API
a_transpose = rv.transpose(a)
```

### Machine Learning Operations

```python
# Create input tensor (channels, height, width)
input_channels = 3
input_height = 32
input_width = 32
input_tensor = np.random.uniform(-1, 1, (input_channels, input_height, input_width)).astype(np.float32)

# Create kernel tensor (num_kernels, channels, height, width)
kernel_num = 16
kernel_height = 3
kernel_width = 3
kernel_tensor = np.random.uniform(-1, 1, (kernel_num, input_channels, kernel_height, kernel_width)).astype(np.float32)

# Convolution operation
stride = (1, 1)
padding = (1, 1)
output = rv.convolution_2d(input_tensor, kernel_tensor, stride[0], stride[1], padding[0], padding[1])
# or using the NumPy-like API
output = rv.conv2d(input_tensor, kernel_tensor, stride=stride, padding=padding)

# Max pooling
pool_size = (2, 2)
stride = (2, 2)
pooled = rv.max_pooling_2d(output, pool_size[0], pool_size[1], stride[0], stride[1])
# or using the NumPy-like API
pooled = rv.max_pool2d(output, kernel_size=pool_size, stride=stride)

# Batch normalization
channels = pooled.shape[0]
gamma = np.random.uniform(0.5, 1.5, channels).astype(np.float32)
beta = np.random.uniform(-0.5, 0.5, channels).astype(np.float32)
mean = np.zeros(channels, dtype=np.float32)
var = np.ones(channels, dtype=np.float32)
epsilon = 1e-5

normalized = rv.batch_norm(pooled, gamma, beta, mean, var, epsilon)

# Softmax
logits = np.random.uniform(-5, 5, 10).astype(np.float32)
probabilities = rv.softmax(logits)
```

## Performance Optimization

### Configuration Options

You can configure the RVV-SIMD library to optimize performance for your specific hardware:

```cpp
// C++
rvv_simd::Config config = rvv_simd::get_default_config();
config.opt_level = rvv_simd::OptimizationLevel::ADVANCED;
config.use_intrinsics = true;
config.enable_profiling = false;
rvv_simd::set_config(config);
```

```python
# Python
# Configuration options are not directly exposed in Python yet
# They are set to optimal defaults at initialization
```

### Memory Layout

For best performance, ensure that your data is aligned and in a contiguous memory layout:

- In C++, allocate aligned memory when possible
- In Python, use `numpy.ascontiguousarray()` if your arrays are not contiguous

### Batch Processing

For multiple small operations, batch them together to reduce overhead:

```python
# Instead of:
for i in range(1000):
    result = rv.vector_add(a[i], b[i])

# Do:
results = []
for i in range(0, 1000, batch_size):
    batch_a = np.stack(a[i:i+batch_size])
    batch_b = np.stack(b[i:i+batch_size])
    batch_results = rv.vector_add(batch_a, batch_b)
    results.append(batch_results)
```

## Benchmarking

The RVV-SIMD library includes benchmarking tools to measure performance:

### Running Benchmarks

```bash
# Build and run benchmarks
cd build
make rvv_simd_benchmarks
./benchmarks/rvv_simd_benchmarks
```

### Interpreting Benchmark Results

The benchmark results show the performance of RVV-SIMD operations compared to equivalent operations on x86 (AVX) and ARM (NEON) architectures. Key metrics include:

- **Throughput**: Operations per second
- **Latency**: Time per operation
- **Speedup**: Relative performance compared to scalar implementation

### Custom Benchmarks

You can create custom benchmarks for your specific use case:

```cpp
#include <benchmark/benchmark.h>
#include "rvv_simd.h"

static void BM_CustomOperation(benchmark::State& state) {
    // Setup
    const size_t size = state.range(0);
    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> result(size);
    
    // Fill with data
    // ...
    
    for (auto _ : state) {
        // Operation to benchmark
        rvv_simd::vector_add(a.data(), b.data(), size, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK(BM_CustomOperation)->Range(1<<10, 1<<20);
BENCHMARK_MAIN();
```

## Troubleshooting

### Common Issues

1. **Library not found**:
   - Ensure the library is installed in a location in your library path
   - Set `LD_LIBRARY_PATH` to include the installation directory

2. **RVV not detected**:
   - Verify that your RISC-V processor supports the Vector extension
   - Check that your compiler has RVV support enabled

3. **Performance issues**:
   - Ensure data is properly aligned and contiguous
   - Check for unnecessary data copies
   - Use appropriate batch sizes for your workload

### Debugging

For debugging issues with the library:

```cpp
// Enable logging
rvv_simd::Config config = rvv_simd::get_default_config();
config.enable_logging = true;
rvv_simd::set_config(config);
```

### Getting Help

If you encounter issues not covered in this guide:

- Check the GitHub issues for similar problems
- Create a new issue with detailed information about your environment and the problem
- Contact the maintainers via the project's communication channels
