#!/usr/bin/env python3
"""
Example demonstrating basic vector operations using RVV-SIMD
"""

import numpy as np
import rvv_simd as rv
import time
import matplotlib.pyplot as plt

def main():
    print("RVV-SIMD Vector Operations Example")
    print("----------------------------------")
    print(f"RVV-SIMD Version: {rv.get_version()}")
    print(f"RVV Support: {'Yes' if rv.is_rvv_supported() else 'No'}")
    print(f"RVV Info: {rv.get_rvv_info()}")
    print()
    
    # Create sample vectors
    size = 1000000
    print(f"Creating vectors with {size} elements...")
    a = np.random.uniform(-10, 10, size).astype(np.float32)
    b = np.random.uniform(-10, 10, size).astype(np.float32)
    
    # Vector addition
    print("\nPerforming vector addition...")
    start_time = time.time()
    c_rv = rv.vector_add(a, b)
    rv_time = time.time() - start_time
    
    start_time = time.time()
    c_np = a + b
    np_time = time.time() - start_time
    
    print(f"RVV-SIMD time: {rv_time:.6f} seconds")
    print(f"NumPy time: {np_time:.6f} seconds")
    print(f"Speedup: {np_time/rv_time:.2f}x")
    print(f"Result verification: {'Passed' if np.allclose(c_rv, c_np) else 'Failed'}")
    
    # Vector dot product
    print("\nPerforming vector dot product...")
    start_time = time.time()
    dot_rv = rv.vector_dot(a, b)
    rv_time = time.time() - start_time
    
    start_time = time.time()
    dot_np = np.dot(a, b)
    np_time = time.time() - start_time
    
    print(f"RVV-SIMD time: {rv_time:.6f} seconds")
    print(f"NumPy time: {np_time:.6f} seconds")
    print(f"Speedup: {np_time/rv_time:.2f}x")
    print(f"Result verification: {'Passed' if np.isclose(dot_rv, dot_np) else 'Failed'}")
    
    # Vector operations
    print("\nPerforming various vector operations...")
    operations = [
        ("Vector Multiplication", rv.vector_mul, lambda x, y: x * y),
        ("Vector Division", rv.vector_div, lambda x, y: x / y),
        ("Vector Scaling", lambda x: rv.vector_scale(x, 2.5), lambda x: x * 2.5),
        ("Vector Normalization", rv.vector_normalize, lambda x: x / np.linalg.norm(x)),
        ("Vector Sigmoid", rv.vector_sigmoid, lambda x: 1 / (1 + np.exp(-x))),
        ("Vector ReLU", rv.vector_relu, lambda x: np.maximum(x, 0)),
    ]
    
    for name, rv_op, np_op in operations:
        print(f"\n{name}...")
        
        if name == "Vector Division" or name == "Vector Multiplication":
            # Binary operations
            start_time = time.time()
            result_rv = rv_op(a, b)
            rv_time = time.time() - start_time
            
            start_time = time.time()
            result_np = np_op(a, b)
            np_time = time.time() - start_time
        else:
            # Unary operations
            start_time = time.time()
            result_rv = rv_op(a)
            rv_time = time.time() - start_time
            
            start_time = time.time()
            result_np = np_op(a)
            np_time = time.time() - start_time
        
        print(f"RVV-SIMD time: {rv_time:.6f} seconds")
        print(f"NumPy time: {np_time:.6f} seconds")
        print(f"Speedup: {np_time/rv_time:.2f}x")
        print(f"Result verification: {'Passed' if np.allclose(result_rv, result_np, rtol=1e-5, atol=1e-5) else 'Failed'}")
    
    # Plot performance comparison
    plot_performance_comparison()

def plot_performance_comparison():
    """
    Plot performance comparison between RVV-SIMD and NumPy for different vector sizes
    """
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    rv_times = []
    np_times = []
    
    for size in sizes:
        a = np.random.uniform(-10, 10, size).astype(np.float32)
        b = np.random.uniform(-10, 10, size).astype(np.float32)
        
        # Measure RVV-SIMD time
        start_time = time.time()
        rv.vector_add(a, b)
        rv_times.append(time.time() - start_time)
        
        # Measure NumPy time
        start_time = time.time()
        a + b
        np_times.append(time.time() - start_time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, rv_times, 'o-', label='RVV-SIMD')
    plt.plot(sizes, np_times, 's-', label='NumPy')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Vector Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: RVV-SIMD vs NumPy')
    plt.legend()
    plt.grid(True)
    plt.savefig('vector_performance_comparison.png')
    print("\nPerformance comparison plot saved as 'vector_performance_comparison.png'")

if __name__ == "__main__":
    main()
