/**
 * @file matrix_ops.h
 * @brief Matrix operations optimized for RISC-V Vector extension
 * 
 * This file contains declarations for matrix operations that are
 * optimized using RISC-V Vector (RVV) instructions.
 */

#ifndef RVV_SIMD_MATRIX_OPS_H
#define RVV_SIMD_MATRIX_OPS_H

#include <stddef.h>

namespace rvv_simd {

/**
 * @brief Add two matrices element-wise
 * 
 * @param a First input matrix (row-major order)
 * @param b Second input matrix (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param result Output matrix (must be pre-allocated)
 * @return Pointer to the result matrix
 */
float* matrix_add(const float* a, const float* b, size_t rows, size_t cols, float* result);

/**
 * @brief Subtract two matrices element-wise
 * 
 * @param a First input matrix (row-major order)
 * @param b Second input matrix (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param result Output matrix (must be pre-allocated)
 * @return Pointer to the result matrix
 */
float* matrix_sub(const float* a, const float* b, size_t rows, size_t cols, float* result);

/**
 * @brief Multiply two matrices element-wise (Hadamard product)
 * 
 * @param a First input matrix (row-major order)
 * @param b Second input matrix (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param result Output matrix (must be pre-allocated)
 * @return Pointer to the result matrix
 */
float* matrix_elem_mul(const float* a, const float* b, size_t rows, size_t cols, float* result);

/**
 * @brief Perform matrix multiplication (a * b)
 * 
 * @param a First input matrix (row-major order)
 * @param b Second input matrix (row-major order)
 * @param a_rows Number of rows in matrix a
 * @param a_cols Number of columns in matrix a / rows in matrix b
 * @param b_cols Number of columns in matrix b
 * @param result Output matrix (must be pre-allocated with dimensions a_rows x b_cols)
 * @return Pointer to the result matrix
 */
float* matrix_mul(const float* a, const float* b, size_t a_rows, size_t a_cols, size_t b_cols, float* result);

/**
 * @brief Transpose a matrix
 * 
 * @param a Input matrix (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param result Output matrix (must be pre-allocated with dimensions cols x rows)
 * @return Pointer to the transposed matrix
 */
float* matrix_transpose(const float* a, size_t rows, size_t cols, float* result);

/**
 * @brief Scale a matrix by a scalar value
 * 
 * @param a Input matrix (row-major order)
 * @param scalar Scalar value
 * @param rows Number of rows
 * @param cols Number of columns
 * @param result Output matrix (must be pre-allocated)
 * @return Pointer to the result matrix
 */
float* matrix_scale(const float* a, float scalar, size_t rows, size_t cols, float* result);

/**
 * @brief Compute the sum of all elements in a matrix
 * 
 * @param a Input matrix (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Sum of all elements
 */
float matrix_sum(const float* a, size_t rows, size_t cols);

/**
 * @brief Apply a function to each element of a matrix
 * 
 * @param a Input matrix (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param func Function to apply (e.g., sigmoid, relu)
 * @param result Output matrix (must be pre-allocated)
 * @return Pointer to the result matrix
 */
float* matrix_apply(const float* a, size_t rows, size_t cols, 
                    float (*func)(float), float* result);

/**
 * @brief Compute the Frobenius norm of a matrix
 * 
 * @param a Input matrix (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Frobenius norm of the matrix
 */
float matrix_norm(const float* a, size_t rows, size_t cols);

} // namespace rvv_simd

#endif // RVV_SIMD_MATRIX_OPS_H
