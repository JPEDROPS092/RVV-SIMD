/**
 * @file vector_ops.h
 * @brief Vector operations optimized for RISC-V Vector extension
 * 
 * This file contains declarations for basic vector operations that are
 * optimized using RISC-V Vector (RVV) instructions.
 */

#ifndef RVV_SIMD_VECTOR_OPS_H
#define RVV_SIMD_VECTOR_OPS_H

#include <stddef.h>

namespace rvv_simd {

/**
 * @brief Add two vectors element-wise
 * 
 * @param a First input vector
 * @param b Second input vector
 * @param length Length of the vectors
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_add(const float* a, const float* b, size_t length, float* result);

/**
 * @brief Subtract two vectors element-wise
 * 
 * @param a First input vector
 * @param b Second input vector
 * @param length Length of the vectors
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_sub(const float* a, const float* b, size_t length, float* result);

/**
 * @brief Multiply two vectors element-wise
 * 
 * @param a First input vector
 * @param b Second input vector
 * @param length Length of the vectors
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_mul(const float* a, const float* b, size_t length, float* result);

/**
 * @brief Divide two vectors element-wise
 * 
 * @param a First input vector
 * @param b Second input vector
 * @param length Length of the vectors
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_div(const float* a, const float* b, size_t length, float* result);

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param a First input vector
 * @param b Second input vector
 * @param length Length of the vectors
 * @return Dot product result
 */
float vector_dot(const float* a, const float* b, size_t length);

/**
 * @brief Scale a vector by a scalar value
 * 
 * @param a Input vector
 * @param scalar Scalar value
 * @param length Length of the vector
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_scale(const float* a, float scalar, size_t length, float* result);

/**
 * @brief Compute the L2 norm (Euclidean norm) of a vector
 * 
 * @param a Input vector
 * @param length Length of the vector
 * @return L2 norm of the vector
 */
float vector_norm(const float* a, size_t length);

/**
 * @brief Normalize a vector to unit length
 * 
 * @param a Input vector
 * @param length Length of the vector
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_normalize(const float* a, size_t length, float* result);

/**
 * @brief Apply exponential function to each element of a vector
 * 
 * @param a Input vector
 * @param length Length of the vector
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_exp(const float* a, size_t length, float* result);

/**
 * @brief Apply natural logarithm to each element of a vector
 * 
 * @param a Input vector
 * @param length Length of the vector
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_log(const float* a, size_t length, float* result);

/**
 * @brief Apply sigmoid function to each element of a vector
 * 
 * @param a Input vector
 * @param length Length of the vector
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_sigmoid(const float* a, size_t length, float* result);

/**
 * @brief Apply tanh function to each element of a vector
 * 
 * @param a Input vector
 * @param length Length of the vector
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_tanh(const float* a, size_t length, float* result);

/**
 * @brief Apply ReLU function to each element of a vector
 * 
 * @param a Input vector
 * @param length Length of the vector
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* vector_relu(const float* a, size_t length, float* result);

} // namespace rvv_simd

#endif // RVV_SIMD_VECTOR_OPS_H
