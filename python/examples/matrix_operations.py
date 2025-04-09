#!/usr/bin/env python3
"""
Example demonstrating matrix operations using RVV-SIMD
"""

import numpy as np
import rvv_simd as rv
import time
import matplotlib.pyplot as plt

def main():
    print("RVV-SIMD Matrix Operations Example")
    print("----------------------------------")
    print(f"RVV-SIMD Version: {rv.get_version()}")
    print(f"RVV Support: {'Yes' if rv.is_rvv_supported() else 'No'}")
    print()
    
    # Create sample matrices
    size = 512
    print(f"Creating {size}x{size} matrices...")
    a = np.random.uniform(-10, 10, (size, size)).astype(np.float32)
    b = np.random.uniform(-10, 10, (size, size)).astype(np.float32)
    
    # Matrix addition
    print("\nPerforming matrix addition...")
    start_time = time.time()
    c_rv = rv.matrix_add(a, b)
    rv_time = time.time() - start_time
    
    start_time = time.time()
    c_np = a + b
    np_time = time.time() - start_time
    
    print(f"RVV-SIMD time: {rv_time:.6f} seconds")
    print(f"NumPy time: {np_time:.6f} seconds")
    print(f"Speedup: {np_time/rv_time:.2f}x")
    print(f"Result verification: {'Passed' if np.allclose(c_rv, c_np) else 'Failed'}")
    
    # Matrix multiplication
    print("\nPerforming matrix multiplication...")
    # Use smaller matrices for matrix multiplication as it's more computationally intensive
    small_size = 128
    a_small = np.random.uniform(-10, 10, (small_size, small_size)).astype(np.float32)
    b_small = np.random.uniform(-10, 10, (small_size, small_size)).astype(np.float32)
    
    start_time = time.time()
    c_rv = rv.matrix_mul(a_small, b_small)
    rv_time = time.time() - start_time
    
    start_time = time.time()
    c_np = np.matmul(a_small, b_small)
    np_time = time.time() - start_time
    
    print(f"RVV-SIMD time: {rv_time:.6f} seconds")
    print(f"NumPy time: {np_time:.6f} seconds")
    print(f"Speedup: {np_time/rv_time:.2f}x")
    print(f"Result verification: {'Passed' if np.allclose(c_rv, c_np, rtol=1e-5, atol=1e-5) else 'Failed'}")
    
    # Matrix operations
    print("\nPerforming various matrix operations...")
    operations = [
        ("Matrix Element-wise Multiplication", rv.matrix_elem_mul, lambda x, y: x * y),
        ("Matrix Transpose", rv.matrix_transpose, lambda x: x.T),
        ("Matrix Scaling", lambda x: rv.matrix_scale(x, 2.5), lambda x: x * 2.5),
    ]
    
    for name, rv_op, np_op in operations:
        print(f"\n{name}...")
        
        if name == "Matrix Element-wise Multiplication":
            # Binary operation
            start_time = time.time()
            result_rv = rv_op(a, b)
            rv_time = time.time() - start_time
            
            start_time = time.time()
            result_np = np_op(a, b)
            np_time = time.time() - start_time
        else:
            # Unary operation
            start_time = time.time()
            result_rv = rv_op(a)
            rv_time = time.time() - start_time
            
            start_time = time.time()
            result_np = np_op(a)
            np_time = time.time() - start_time
        
        print(f"RVV-SIMD time: {rv_time:.6f} seconds")
        print(f"NumPy time: {np_time:.6f} seconds")
        print(f"Speedup: {np_time/rv_time:.2f}x")
        print(f"Result verification: {'Passed' if np.allclose(result_rv, result_np) else 'Failed'}")
    
    # Plot performance comparison for matrix multiplication
    plot_matmul_performance_comparison()

def plot_matmul_performance_comparison():
    """
    Plot performance comparison between RVV-SIMD and NumPy for matrix multiplication
    with different matrix sizes
    """
    sizes = [16, 32, 64, 128, 256]
    rv_times = []
    np_times = []
    
    for size in sizes:
        a = np.random.uniform(-10, 10, (size, size)).astype(np.float32)
        b = np.random.uniform(-10, 10, (size, size)).astype(np.float32)
        
        # Measure RVV-SIMD time
        start_time = time.time()
        rv.matrix_mul(a, b)
        rv_times.append(time.time() - start_time)
        
        # Measure NumPy time
        start_time = time.time()
        np.matmul(a, b)
        np_times.append(time.time() - start_time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, rv_times, 'o-', label='RVV-SIMD')
    plt.plot(sizes, np_times, 's-', label='NumPy')
    plt.xlabel('Matrix Size (N×N)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Matrix Multiplication Performance: RVV-SIMD vs NumPy')
    plt.legend()
    plt.grid(True)
    plt.savefig('matrix_performance_comparison.png')
    print("\nPerformance comparison plot saved as 'matrix_performance_comparison.png'")

if __name__ == "__main__":
    main()
