/**
 * @file ml_ops.h
 * @brief Machine learning operations optimized for RISC-V Vector extension
 * 
 * This file contains declarations for machine learning specific operations
 * that are optimized using RISC-V Vector (RVV) instructions.
 */

#ifndef RVV_SIMD_ML_OPS_H
#define RVV_SIMD_ML_OPS_H

#include <stddef.h>

namespace rvv_simd {

/**
 * @brief Perform convolution operation for CNN
 * 
 * @param input Input tensor (row-major order)
 * @param kernel Convolution kernel (row-major order)
 * @param input_h Input height
 * @param input_w Input width
 * @param input_c Input channels
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param kernel_c Kernel channels (must match input_c)
 * @param kernel_n Number of kernels (output channels)
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param padding_h Padding height
 * @param padding_w Padding width
 * @param result Output tensor (must be pre-allocated)
 * @return Pointer to the result tensor
 */
float* convolution_2d(const float* input, const float* kernel,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t kernel_h, size_t kernel_w, size_t kernel_c, size_t kernel_n,
                     size_t stride_h, size_t stride_w,
                     size_t padding_h, size_t padding_w,
                     float* result);

/**
 * @brief Perform max pooling operation
 * 
 * @param input Input tensor (row-major order)
 * @param input_h Input height
 * @param input_w Input width
 * @param input_c Input channels
 * @param pool_h Pooling window height
 * @param pool_w Pooling window width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param result Output tensor (must be pre-allocated)
 * @return Pointer to the result tensor
 */
float* max_pooling_2d(const float* input,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t pool_h, size_t pool_w,
                     size_t stride_h, size_t stride_w,
                     float* result);

/**
 * @brief Perform average pooling operation
 * 
 * @param input Input tensor (row-major order)
 * @param input_h Input height
 * @param input_w Input width
 * @param input_c Input channels
 * @param pool_h Pooling window height
 * @param pool_w Pooling window width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param result Output tensor (must be pre-allocated)
 * @return Pointer to the result tensor
 */
float* avg_pooling_2d(const float* input,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t pool_h, size_t pool_w,
                     size_t stride_h, size_t stride_w,
                     float* result);

/**
 * @brief Apply batch normalization
 * 
 * @param input Input tensor
 * @param gamma Scale parameter
 * @param beta Shift parameter
 * @param mean Mean values
 * @param var Variance values
 * @param epsilon Small constant for numerical stability
 * @param size Number of elements
 * @param channels Number of channels
 * @param result Output tensor (must be pre-allocated)
 * @return Pointer to the result tensor
 */
float* batch_norm(const float* input, const float* gamma, const float* beta,
                 const float* mean, const float* var, float epsilon,
                 size_t size, size_t channels, float* result);

/**
 * @brief Apply softmax function to a vector
 * 
 * @param input Input vector
 * @param size Vector size
 * @param result Output vector (must be pre-allocated)
 * @return Pointer to the result vector
 */
float* softmax(const float* input, size_t size, float* result);

/**
 * @brief Compute cross-entropy loss
 * 
 * @param predictions Predicted probabilities
 * @param targets Target values (one-hot encoded)
 * @param batch_size Batch size
 * @param num_classes Number of classes
 * @return Cross-entropy loss value
 */
float cross_entropy_loss(const float* predictions, const float* targets,
                        size_t batch_size, size_t num_classes);

/**
 * @brief Compute gradients for backpropagation
 * 
 * @param output_grad Output gradients
 * @param input Input values
 * @param size Number of elements
 * @param result Gradient result (must be pre-allocated)
 * @return Pointer to the result gradients
 */
float* compute_gradients(const float* output_grad, const float* input,
                        size_t size, float* result);

/**
 * @brief Apply dropout during training
 * 
 * @param input Input tensor
 * @param mask Dropout mask (0 for dropped, 1 for kept)
 * @param size Number of elements
 * @param dropout_rate Dropout rate (0.0 to 1.0)
 * @param result Output tensor (must be pre-allocated)
 * @return Pointer to the result tensor
 */
float* apply_dropout(const float* input, unsigned char* mask,
                    size_t size, float dropout_rate, float* result);

} // namespace rvv_simd

#endif // RVV_SIMD_ML_OPS_H
