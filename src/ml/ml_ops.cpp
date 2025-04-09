#include "rvv_simd/ml_ops.h"
#include "rvv_simd/vector_ops.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// Check if we can use RVV intrinsics
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rvv_simd {

float* convolution_2d(const float* input, const float* kernel,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t kernel_h, size_t kernel_w, size_t kernel_c, size_t kernel_n,
                     size_t stride_h, size_t stride_w,
                     size_t padding_h, size_t padding_w,
                     float* result) {
    // Calculate output dimensions
    size_t output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
    size_t output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    // Initialize result to zeros
    memset(result, 0, output_h * output_w * kernel_n * sizeof(float));
    
    // Perform convolution
    for (size_t n = 0; n < kernel_n; n++) {
        for (size_t oh = 0; oh < output_h; oh++) {
            for (size_t ow = 0; ow < output_w; ow++) {
                float sum = 0.0f;
                
                for (size_t c = 0; c < input_c; c++) {
                    for (size_t kh = 0; kh < kernel_h; kh++) {
                        for (size_t kw = 0; kw < kernel_w; kw++) {
                            int ih = oh * stride_h + kh - padding_h;
                            int iw = ow * stride_w + kw - padding_w;
                            
                            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                float input_val = input[(c * input_h + ih) * input_w + iw];
                                float kernel_val = kernel[((n * kernel_c + c) * kernel_h + kh) * kernel_w + kw];
                                sum += input_val * kernel_val;
                            }
                        }
                    }
                }
                
                result[(n * output_h + oh) * output_w + ow] = sum;
            }
        }
    }
    
    return result;
}

float* max_pooling_2d(const float* input,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t pool_h, size_t pool_w,
                     size_t stride_h, size_t stride_w,
                     float* result) {
    // Calculate output dimensions
    size_t output_h = (input_h - pool_h) / stride_h + 1;
    size_t output_w = (input_w - pool_w) / stride_w + 1;
    
    // Perform max pooling
    for (size_t c = 0; c < input_c; c++) {
        for (size_t oh = 0; oh < output_h; oh++) {
            for (size_t ow = 0; ow < output_w; ow++) {
                float max_val = -INFINITY;
                
                for (size_t ph = 0; ph < pool_h; ph++) {
                    for (size_t pw = 0; pw < pool_w; pw++) {
                        size_t ih = oh * stride_h + ph;
                        size_t iw = ow * stride_w + pw;
                        
                        if (ih < input_h && iw < input_w) {
                            float val = input[(c * input_h + ih) * input_w + iw];
                            max_val = std::max(max_val, val);
                        }
                    }
                }
                
                result[(c * output_h + oh) * output_w + ow] = max_val;
            }
        }
    }
    
    return result;
}

float* avg_pooling_2d(const float* input,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t pool_h, size_t pool_w,
                     size_t stride_h, size_t stride_w,
                     float* result) {
    // Calculate output dimensions
    size_t output_h = (input_h - pool_h) / stride_h + 1;
    size_t output_w = (input_w - pool_w) / stride_w + 1;
    
    // Perform average pooling
    for (size_t c = 0; c < input_c; c++) {
        for (size_t oh = 0; oh < output_h; oh++) {
            for (size_t ow = 0; ow < output_w; ow++) {
                float sum = 0.0f;
                size_t count = 0;
                
                for (size_t ph = 0; ph < pool_h; ph++) {
                    for (size_t pw = 0; pw < pool_w; pw++) {
                        size_t ih = oh * stride_h + ph;
                        size_t iw = ow * stride_w + pw;
                        
                        if (ih < input_h && iw < input_w) {
                            sum += input[(c * input_h + ih) * input_w + iw];
                            count++;
                        }
                    }
                }
                
                result[(c * output_h + oh) * output_w + ow] = (count > 0) ? (sum / count) : 0.0f;
            }
        }
    }
    
    return result;
}

float* batch_norm(const float* input, const float* gamma, const float* beta,
                 const float* mean, const float* var, float epsilon,
                 size_t size, size_t channels, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for batch normalization
    for (size_t c = 0; c < channels; c++) {
        float scale = gamma[c] / std::sqrt(var[c] + epsilon);
        float shift = beta[c] - scale * mean[c];
        
        size_t vl;
        for (size_t i = 0; i < size; i += vl) {
            vl = __riscv_vsetvl_e32m8(size - i);
            vfloat32m8_t vinput = __riscv_vle32_v_f32m8(&input[c * size + i], vl);
            vfloat32m8_t vscale = __riscv_vfmv_v_f_f32m8(scale, vl);
            vfloat32m8_t vshift = __riscv_vfmv_v_f_f32m8(shift, vl);
            
            // normalized = gamma * (input - mean) / sqrt(var + epsilon) + beta
            //            = scale * input + shift
            vfloat32m8_t vscaled = __riscv_vfmul_vv_f32m8(vinput, vscale, vl);
            vfloat32m8_t vresult = __riscv_vfadd_vv_f32m8(vscaled, vshift, vl);
            
            __riscv_vse32_v_f32m8(&result[c * size + i], vresult, vl);
        }
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t c = 0; c < channels; c++) {
        float scale = gamma[c] / std::sqrt(var[c] + epsilon);
        float shift = beta[c] - scale * mean[c];
        
        for (size_t i = 0; i < size; i++) {
            result[c * size + i] = scale * input[c * size + i] + shift;
        }
    }
#endif
    
    return result;
}

float* softmax(const float* input, size_t size, float* result) {
    // Find the maximum value for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp(x - max) for each element
    float sum = 0.0f;
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for softmax computation
    size_t vl;
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, 1);
    
    for (size_t i = 0; i < size; i += vl) {
        vl = __riscv_vsetvl_e32m8(size - i);
        vfloat32m8_t vinput = __riscv_vle32_v_f32m8(&input[i], vl);
        vfloat32m8_t vmax = __riscv_vfmv_v_f_f32m8(max_val, vl);
        vfloat32m8_t vdiff = __riscv_vfsub_vv_f32m8(vinput, vmax, vl);
        
        // Compute exp(x - max)
        #if defined(__riscv_vfexp)
        vfloat32m8_t vexp = __riscv_vfexp_v_f32m8(vdiff, vl);
        #else
        // Fallback when vfexp is not available
        vfloat32m8_t vexp;
        float* exp_values = new float[vl];
        __riscv_vse32_v_f32m8(exp_values, vdiff, vl);
        for (size_t j = 0; j < vl; j++) {
            exp_values[j] = std::exp(exp_values[j]);
        }
        vexp = __riscv_vle32_v_f32m8(exp_values, vl);
        delete[] exp_values;
        #endif
        
        // Store intermediate exp values
        __riscv_vse32_v_f32m8(&result[i], vexp, vl);
        
        // Accumulate sum
        vsum = __riscv_vfredusum_vs_f32m8_f32m1(vexp, vsum, vl);
    }
    
    sum = __riscv_vfmv_f_s_f32m1_f32(vsum);
    
    // Normalize by sum
    for (size_t i = 0; i < size; i += vl) {
        vl = __riscv_vsetvl_e32m8(size - i);
        vfloat32m8_t vexp = __riscv_vle32_v_f32m8(&result[i], vl);
        vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(sum, vl);
        vfloat32m8_t vresult = __riscv_vfdiv_vv_f32m8(vexp, vsum, vl);
        __riscv_vse32_v_f32m8(&result[i], vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < size; i++) {
        result[i] = std::exp(input[i] - max_val);
        sum += result[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
#endif
    
    return result;
}

float cross_entropy_loss(const float* predictions, const float* targets,
                        size_t batch_size, size_t num_classes) {
    float loss = 0.0f;
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            // Add a small epsilon to avoid log(0)
            float pred = std::max(predictions[idx], 1e-7f);
            loss -= targets[idx] * std::log(pred);
        }
    }
    
    return loss / batch_size;
}

float* compute_gradients(const float* output_grad, const float* input,
                        size_t size, float* result) {
    // This is a simplified gradient computation
    // In a real implementation, this would depend on the specific operation
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for gradient computation
    size_t vl;
    for (size_t i = 0; i < size; i += vl) {
        vl = __riscv_vsetvl_e32m8(size - i);
        vfloat32m8_t voutput_grad = __riscv_vle32_v_f32m8(&output_grad[i], vl);
        vfloat32m8_t vinput = __riscv_vle32_v_f32m8(&input[i], vl);
        
        // Simple gradient computation (depends on the actual operation)
        vfloat32m8_t vresult = __riscv_vfmul_vv_f32m8(voutput_grad, vinput, vl);
        
        __riscv_vse32_v_f32m8(&result[i], vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < size; i++) {
        result[i] = output_grad[i] * input[i];
    }
#endif
    
    return result;
}

float* apply_dropout(const float* input, unsigned char* mask,
                    size_t size, float dropout_rate, float* result) {
    float scale = 1.0f / (1.0f - dropout_rate);
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for dropout application
    size_t vl;
    for (size_t i = 0; i < size; i += vl) {
        vl = __riscv_vsetvl_e32m8(size - i);
        vfloat32m8_t vinput = __riscv_vle32_v_f32m8(&input[i], vl);
        
        // Load mask values
        vbool4_t vmask = __riscv_vlm_v_b4(mask + i, vl);
        
        // Apply mask and scale
        vfloat32m8_t vscale = __riscv_vfmv_v_f_f32m8(scale, vl);
        vfloat32m8_t vzero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t vscaled = __riscv_vfmul_vv_f32m8(vinput, vscale, vl);
        vfloat32m8_t vresult = __riscv_vmerge_vvm_f32m8(vzero, vscaled, vmask, vl);
        
        __riscv_vse32_v_f32m8(&result[i], vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < size; i++) {
        result[i] = mask[i] ? (input[i] * scale) : 0.0f;
    }
#endif
    
    return result;
}

} // namespace rvv_simd
