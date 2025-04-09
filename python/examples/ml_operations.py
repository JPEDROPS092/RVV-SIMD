#!/usr/bin/env python3
"""
Example demonstrating machine learning operations using RVV-SIMD
"""

import numpy as np
import rvv_simd as rv
import time
import matplotlib.pyplot as plt

def main():
    print("RVV-SIMD Machine Learning Operations Example")
    print("-------------------------------------------")
    print(f"RVV-SIMD Version: {rv.get_version()}")
    print(f"RVV Support: {'Yes' if rv.is_rvv_supported() else 'No'}")
    print()
    
    # Convolution example
    print("\nPerforming 2D Convolution...")
    # Create a sample input tensor (channels, height, width)
    input_channels = 3
    input_height = 32
    input_width = 32
    input_tensor = np.random.uniform(-1, 1, (input_channels, input_height, input_width)).astype(np.float32)
    
    # Create a sample kernel tensor (num_kernels, channels, height, width)
    kernel_num = 16
    kernel_height = 3
    kernel_width = 3
    kernel_tensor = np.random.uniform(-1, 1, (kernel_num, input_channels, kernel_height, kernel_width)).astype(np.float32)
    
    # Set convolution parameters
    stride = (1, 1)
    padding = (1, 1)
    
    # Perform convolution using RVV-SIMD
    start_time = time.time()
    output_rv = rv.convolution_2d(input_tensor, kernel_tensor, stride[0], stride[1], padding[0], padding[1])
    rv_time = time.time() - start_time
    print(f"RVV-SIMD convolution time: {rv_time:.6f} seconds")
    print(f"Output shape: {output_rv.shape}")
    
    # Max pooling example
    print("\nPerforming Max Pooling...")
    pool_size = (2, 2)
    stride = (2, 2)
    
    # Perform max pooling using RVV-SIMD
    start_time = time.time()
    pooled_rv = rv.max_pooling_2d(output_rv, pool_size[0], pool_size[1], stride[0], stride[1])
    rv_time = time.time() - start_time
    print(f"RVV-SIMD max pooling time: {rv_time:.6f} seconds")
    print(f"Pooled output shape: {pooled_rv.shape}")
    
    # Batch normalization example
    print("\nPerforming Batch Normalization...")
    # Create parameters for batch normalization
    channels = pooled_rv.shape[0]
    gamma = np.random.uniform(0.5, 1.5, channels).astype(np.float32)
    beta = np.random.uniform(-0.5, 0.5, channels).astype(np.float32)
    mean = np.zeros(channels, dtype=np.float32)
    var = np.ones(channels, dtype=np.float32)
    epsilon = 1e-5
    
    # Perform batch normalization using RVV-SIMD
    start_time = time.time()
    normalized_rv = rv.batch_norm(pooled_rv, gamma, beta, mean, var, epsilon)
    rv_time = time.time() - start_time
    print(f"RVV-SIMD batch normalization time: {rv_time:.6f} seconds")
    
    # Softmax example
    print("\nPerforming Softmax...")
    # Create a sample logits vector
    logits = np.random.uniform(-5, 5, 1000).astype(np.float32)
    
    # Perform softmax using RVV-SIMD
    start_time = time.time()
    probs_rv = rv.softmax(logits)
    rv_time = time.time() - start_time
    
    # Compare with NumPy implementation
    def softmax_np(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    start_time = time.time()
    probs_np = softmax_np(logits)
    np_time = time.time() - start_time
    
    print(f"RVV-SIMD softmax time: {rv_time:.6f} seconds")
    print(f"NumPy softmax time: {np_time:.6f} seconds")
    print(f"Speedup: {np_time/rv_time:.2f}x")
    print(f"Result verification: {'Passed' if np.allclose(probs_rv, probs_np, rtol=1e-5, atol=1e-5) else 'Failed'}")
    print(f"Sum of probabilities: {np.sum(probs_rv):.6f} (should be close to 1.0)")
    
    # Simple neural network forward pass example
    print("\nSimulating a Simple Neural Network Forward Pass...")
    # Create a simple 2-layer neural network
    input_size = 784  # e.g., flattened MNIST image
    hidden_size = 128
    output_size = 10  # 10 classes
    
    # Create random weights and biases
    W1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_size)).astype(np.float32)
    b1 = np.random.uniform(-0.1, 0.1, hidden_size).astype(np.float32)
    W2 = np.random.uniform(-0.1, 0.1, (hidden_size, output_size)).astype(np.float32)
    b2 = np.random.uniform(-0.1, 0.1, output_size).astype(np.float32)
    
    # Create a batch of random input data
    batch_size = 32
    X = np.random.uniform(-1, 1, (batch_size, input_size)).astype(np.float32)
    
    # Forward pass using RVV-SIMD
    start_time = time.time()
    
    # First layer: X @ W1 + b1, then ReLU
    hidden = np.zeros((batch_size, hidden_size), dtype=np.float32)
    for i in range(batch_size):
        hidden[i] = rv.vector_add(rv.matrix_mul(X[i:i+1], W1), b1)
    hidden_relu = np.zeros_like(hidden)
    for i in range(batch_size):
        hidden_relu[i] = rv.vector_relu(hidden[i])
    
    # Second layer: hidden_relu @ W2 + b2, then softmax
    logits = np.zeros((batch_size, output_size), dtype=np.float32)
    for i in range(batch_size):
        logits[i] = rv.vector_add(rv.matrix_mul(hidden_relu[i:i+1], W2), b2)
    
    # Apply softmax to each example in the batch
    probs = np.zeros_like(logits)
    for i in range(batch_size):
        probs[i] = rv.softmax(logits[i])
    
    rv_time = time.time() - start_time
    print(f"RVV-SIMD neural network forward pass time: {rv_time:.6f} seconds")
    
    # Plot performance comparison for different ML operations
    plot_ml_performance_comparison()

def plot_ml_performance_comparison():
    """
    Plot performance comparison for different ML operations
    """
    # Define the operations to benchmark
    operations = [
        "Convolution",
        "Max Pooling",
        "Batch Norm",
        "Softmax",
        "Matrix Mul"
    ]
    
    # Sizes for each operation
    sizes = [
        [16, 32, 64],  # Input sizes for convolution
        [16, 32, 64],  # Input sizes for max pooling
        [16, 32, 64],  # Channel sizes for batch norm
        [1000, 5000, 10000],  # Vector sizes for softmax
        [64, 128, 256]  # Matrix sizes for matrix multiplication
    ]
    
    # Initialize time arrays
    rv_times = [[] for _ in range(len(operations))]
    
    # Benchmark convolution
    for size in sizes[0]:
        input_tensor = np.random.uniform(-1, 1, (3, size, size)).astype(np.float32)
        kernel_tensor = np.random.uniform(-1, 1, (16, 3, 3, 3)).astype(np.float32)
        
        start_time = time.time()
        rv.convolution_2d(input_tensor, kernel_tensor, 1, 1, 1, 1)
        rv_times[0].append(time.time() - start_time)
    
    # Benchmark max pooling
    for size in sizes[1]:
        input_tensor = np.random.uniform(-1, 1, (16, size, size)).astype(np.float32)
        
        start_time = time.time()
        rv.max_pooling_2d(input_tensor, 2, 2, 2, 2)
        rv_times[1].append(time.time() - start_time)
    
    # Benchmark batch normalization
    for size in sizes[2]:
        input_tensor = np.random.uniform(-1, 1, (size, 32, 32)).astype(np.float32)
        gamma = np.random.uniform(0.5, 1.5, size).astype(np.float32)
        beta = np.random.uniform(-0.5, 0.5, size).astype(np.float32)
        mean = np.zeros(size, dtype=np.float32)
        var = np.ones(size, dtype=np.float32)
        
        start_time = time.time()
        rv.batch_norm(input_tensor, gamma, beta, mean, var, 1e-5)
        rv_times[2].append(time.time() - start_time)
    
    # Benchmark softmax
    for size in sizes[3]:
        logits = np.random.uniform(-5, 5, size).astype(np.float32)
        
        start_time = time.time()
        rv.softmax(logits)
        rv_times[3].append(time.time() - start_time)
    
    # Benchmark matrix multiplication
    for size in sizes[4]:
        a = np.random.uniform(-1, 1, (size, size)).astype(np.float32)
        b = np.random.uniform(-1, 1, (size, size)).astype(np.float32)
        
        start_time = time.time()
        rv.matrix_mul(a, b)
        rv_times[4].append(time.time() - start_time)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    for i, op in enumerate(operations):
        plt.subplot(2, 3, i+1)
        plt.plot(sizes[i], rv_times[i], 'o-', label='RVV-SIMD')
        plt.xlabel('Size')
        plt.ylabel('Time (seconds)')
        plt.title(f'{op} Performance')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ml_performance_comparison.png')
    print("\nML performance comparison plot saved as 'ml_performance_comparison.png'")

if __name__ == "__main__":
    main()
