#include "rvv_simd/matrix_ops.h"
#include "rvv_simd/vector_ops.h"
#include <cmath>
#include <cstring>

// Check if we can use RVV intrinsics
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rvv_simd {

float* matrix_add(const float* a, const float* b, size_t rows, size_t cols, float* result) {
    // Matrix addition is element-wise, so we can use vector_add
    return vector_add(a, b, rows * cols, result);
}

float* matrix_sub(const float* a, const float* b, size_t rows, size_t cols, float* result) {
    // Matrix subtraction is element-wise, so we can use vector_sub
    return vector_sub(a, b, rows * cols, result);
}

float* matrix_elem_mul(const float* a, const float* b, size_t rows, size_t cols, float* result) {
    // Element-wise multiplication, so we can use vector_mul
    return vector_mul(a, b, rows * cols, result);
}

float* matrix_mul(const float* a, const float* b, size_t a_rows, size_t a_cols, size_t b_cols, float* result) {
    // Initialize result matrix to zeros
    memset(result, 0, a_rows * b_cols * sizeof(float));
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for matrix multiplication
    for (size_t i = 0; i < a_rows; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            size_t vl;
            vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, 1);
            
            for (size_t k = 0; k < a_cols; k += vl) {
                vl = __riscv_vsetvl_e32m8(a_cols - k);
                vfloat32m8_t va = __riscv_vle32_v_f32m8(&a[i * a_cols + k], vl);
                
                // For each element in the current chunk, multiply by the corresponding b element
                for (size_t l = 0; l < vl; l++) {
                    if (k + l < a_cols) {
                        float a_val = a[i * a_cols + k + l];
                        float b_val = b[(k + l) * b_cols + j];
                        result[i * b_cols + j] += a_val * b_val;
                    }
                }
            }
        }
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < a_rows; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a_cols; k++) {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            result[i * b_cols + j] = sum;
        }
    }
#endif
    
    return result;
}

float* matrix_transpose(const float* a, size_t rows, size_t cols, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for matrix transpose
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j += 1) {
            result[j * rows + i] = a[i * cols + j];
        }
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j * rows + i] = a[i * cols + j];
        }
    }
#endif
    
    return result;
}

float* matrix_scale(const float* a, float scalar, size_t rows, size_t cols, float* result) {
    // Matrix scaling is element-wise, so we can use vector_scale
    return vector_scale(a, scalar, rows * cols, result);
}

float matrix_sum(const float* a, size_t rows, size_t cols) {
    float sum = 0.0f;
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for computing the sum
    size_t vl;
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, 1);
    
    for (size_t i = 0; i < rows * cols; i += vl) {
        vl = __riscv_vsetvl_e32m8(rows * cols - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vsum = __riscv_vfredusum_vs_f32m8_f32m1(va, vsum, vl);
    }
    
    sum = __riscv_vfmv_f_s_f32m1_f32(vsum);
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < rows * cols; i++) {
        sum += a[i];
    }
#endif
    
    return sum;
}

float* matrix_apply(const float* a, size_t rows, size_t cols, float (*func)(float), float* result) {
#if defined(__riscv_vector)
    // For RVV platforms, we need to apply the function element by element
    // since we can't directly pass a function pointer to RVV intrinsics
    for (size_t i = 0; i < rows * cols; i++) {
        result[i] = func(a[i]);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < rows * cols; i++) {
        result[i] = func(a[i]);
    }
#endif
    
    return result;
}

float matrix_norm(const float* a, size_t rows, size_t cols) {
    float sum_squares = 0.0f;
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for computing the Frobenius norm
    size_t vl;
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, 1);
    
    for (size_t i = 0; i < rows * cols; i += vl) {
        vl = __riscv_vsetvl_e32m8(rows * cols - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vsquare = __riscv_vfmul_vv_f32m8(va, va, vl);
        vsum = __riscv_vfredusum_vs_f32m8_f32m1(vsquare, vsum, vl);
    }
    
    sum_squares = __riscv_vfmv_f_s_f32m1_f32(vsum);
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < rows * cols; i++) {
        sum_squares += a[i] * a[i];
    }
#endif
    
    return std::sqrt(sum_squares);
}

} // namespace rvv_simd
