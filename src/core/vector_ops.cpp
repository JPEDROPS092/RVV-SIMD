#include "rvv_simd/vector_ops.h"
#include "rvv_simd/config.h"
#include <cmath>
#include <cstring>

// Check if we can use RVV intrinsics
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rvv_simd {

float* vector_add(const float* a, const float* b, size_t length, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for vector addition
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vresult = __riscv_vfadd_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result[i] = a[i] + b[i];
    }
#endif
    return result;
}

float* vector_sub(const float* a, const float* b, size_t length, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for vector subtraction
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vresult = __riscv_vfsub_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result[i] = a[i] - b[i];
    }
#endif
    return result;
}

float* vector_mul(const float* a, const float* b, size_t length, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for vector multiplication
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vresult = __riscv_vfmul_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result[i] = a[i] * b[i];
    }
#endif
    return result;
}

float* vector_div(const float* a, const float* b, size_t length, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for vector division
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vresult = __riscv_vfdiv_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result[i] = a[i] / b[i];
    }
#endif
    return result;
}

float vector_dot(const float* a, const float* b, size_t length) {
    float result = 0.0f;
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for dot product
    size_t vl;
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, 1);
    
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vmul = __riscv_vfmul_vv_f32m8(va, vb, vl);
        vsum = __riscv_vfredusum_vs_f32m8_f32m1(vmul, vsum, vl);
    }
    
    result = __riscv_vfmv_f_s_f32m1_f32(vsum);
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
#endif
    
    return result;
}

float* vector_scale(const float* a, float scalar, size_t length, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for scalar multiplication
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vresult = __riscv_vfmul_vf_f32m8(va, scalar, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result[i] = a[i] * scalar;
    }
#endif
    return result;
}

float vector_norm(const float* a, size_t length) {
    float sum_squares = 0.0f;
    
#if defined(__riscv_vector)
    // Using RVV intrinsics for computing the norm
    size_t vl;
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, 1);
    
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vsquare = __riscv_vfmul_vv_f32m8(va, va, vl);
        vsum = __riscv_vfredusum_vs_f32m8_f32m1(vsquare, vsum, vl);
    }
    
    sum_squares = __riscv_vfmv_f_s_f32m1_f32(vsum);
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        sum_squares += a[i] * a[i];
    }
#endif
    
    return std::sqrt(sum_squares);
}

float* vector_normalize(const float* a, size_t length, float* result) {
    float norm = vector_norm(a, length);
    
    if (norm < 1e-10f) {
        // Avoid division by zero
        memset(result, 0, length * sizeof(float));
        return result;
    }
    
    return vector_scale(a, 1.0f / norm, length, result);
}

float* vector_exp(const float* a, size_t length, float* result) {
#if defined(__riscv_vector) && defined(__riscv_vfexp)
    // Using RVV intrinsics for exponential if available
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vresult = __riscv_vfexp_v_f32m8(va, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms or when vfexp is not available
    for (size_t i = 0; i < length; i++) {
        result[i] = std::exp(a[i]);
    }
#endif
    return result;
}

float* vector_log(const float* a, size_t length, float* result) {
#if defined(__riscv_vector) && defined(__riscv_vflog)
    // Using RVV intrinsics for logarithm if available
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vresult = __riscv_vflog_v_f32m8(va, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms or when vflog is not available
    for (size_t i = 0; i < length; i++) {
        result[i] = std::log(a[i]);
    }
#endif
    return result;
}

float* vector_sigmoid(const float* a, size_t length, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for sigmoid computation
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        
        // Compute -x for stability with large positive values
        vfloat32m8_t vneg = __riscv_vfneg_v_f32m8(va, vl);
        
        // Compute exp(-x)
        #if defined(__riscv_vfexp)
        vfloat32m8_t vexp = __riscv_vfexp_v_f32m8(vneg, vl);
        #else
        // Fallback when vfexp is not available
        vfloat32m8_t vexp;
        float* exp_values = new float[vl];
        __riscv_vse32_v_f32m8(exp_values, vneg, vl);
        for (size_t j = 0; j < vl; j++) {
            exp_values[j] = std::exp(exp_values[j]);
        }
        vexp = __riscv_vle32_v_f32m8(exp_values, vl);
        delete[] exp_values;
        #endif
        
        // Compute 1 + exp(-x)
        vfloat32m8_t vone = __riscv_vfmv_v_f_f32m8(1.0f, vl);
        vfloat32m8_t vsum = __riscv_vfadd_vv_f32m8(vone, vexp, vl);
        
        // Compute 1 / (1 + exp(-x))
        vfloat32m8_t vresult = __riscv_vfdiv_vv_f32m8(vone, vsum, vl);
        
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
#endif
    return result;
}

float* vector_tanh(const float* a, size_t length, float* result) {
#if defined(__riscv_vector) && defined(__riscv_vftanh)
    // Using RVV intrinsics for tanh if available
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vresult = __riscv_vftanh_v_f32m8(va, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms or when vftanh is not available
    for (size_t i = 0; i < length; i++) {
        result[i] = std::tanh(a[i]);
    }
#endif
    return result;
}

float* vector_relu(const float* a, size_t length, float* result) {
#if defined(__riscv_vector)
    // Using RVV intrinsics for ReLU computation
    size_t vl;
    for (size_t i = 0; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vzero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t vresult = __riscv_vfmax_vv_f32m8(va, vzero, vl);
        __riscv_vse32_v_f32m8(result + i, vresult, vl);
    }
#else
    // Fallback implementation for non-RVV platforms
    for (size_t i = 0; i < length; i++) {
        result[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
    }
#endif
    return result;
}

} // namespace rvv_simd
