#include <benchmark/benchmark.h>
#include "rvv_simd.h"
#include <vector>
#include <random>
#include <cmath>
#include <string>

// Architecture-specific headers
#if defined(__x86_64__)
#include <immintrin.h>  // AVX/AVX2
#elif defined(__arm__)
#include <arm_neon.h>   // NEON
#endif

// Helper function to generate random data
std::vector<float> generate_random_data(size_t size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
    
    return data;
}

// Native implementation for x86 using AVX
#if defined(__x86_64__) && defined(__AVX__)
void vector_add_avx(const float* a, const float* b, size_t length, float* result) {
    const size_t step = 8;  // AVX processes 8 floats at a time
    
    for (size_t i = 0; i < length; i += step) {
        size_t remaining = std::min(step, length - i);
        
        if (remaining == step) {
            // Full AVX register
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vresult = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(result + i, vresult);
        } else {
            // Handle remaining elements
            for (size_t j = 0; j < remaining; j++) {
                result[i + j] = a[i + j] + b[i + j];
            }
        }
    }
}

void matrix_mul_avx(const float* a, const float* b, size_t a_rows, size_t a_cols, size_t b_cols, float* result) {
    // Initialize result matrix to zeros
    std::memset(result, 0, a_rows * b_cols * sizeof(float));
    
    for (size_t i = 0; i < a_rows; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t k = 0; k < a_cols; k += 8) {
                size_t remaining = std::min(size_t(8), a_cols - k);
                
                if (remaining == 8) {
                    // Load 8 elements from row i of matrix a
                    __m256 va = _mm256_loadu_ps(&a[i * a_cols + k]);
                    
                    // Load 8 elements from column j of matrix b
                    float temp[8];
                    for (int l = 0; l < 8; l++) {
                        temp[l] = b[(k + l) * b_cols + j];
                    }
                    __m256 vb = _mm256_loadu_ps(temp);
                    
                    // Multiply and accumulate
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
                } else {
                    // Handle remaining elements
                    for (size_t l = 0; l < remaining; l++) {
                        result[i * b_cols + j] += a[i * a_cols + k + l] * b[(k + l) * b_cols + j];
                    }
                }
            }
            
            // Horizontal sum of the AVX register
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            for (int l = 0; l < 8; l++) {
                result[i * b_cols + j] += temp[l];
            }
        }
    }
}
#endif

// Native implementation for ARM using NEON
#if defined(__arm__) && defined(__ARM_NEON)
void vector_add_neon(const float* a, const float* b, size_t length, float* result) {
    const size_t step = 4;  // NEON processes 4 floats at a time
    
    for (size_t i = 0; i < length; i += step) {
        size_t remaining = std::min(step, length - i);
        
        if (remaining == step) {
            // Full NEON register
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vresult = vaddq_f32(va, vb);
            vst1q_f32(result + i, vresult);
        } else {
            // Handle remaining elements
            for (size_t j = 0; j < remaining; j++) {
                result[i + j] = a[i + j] + b[i + j];
            }
        }
    }
}

void matrix_mul_neon(const float* a, const float* b, size_t a_rows, size_t a_cols, size_t b_cols, float* result) {
    // Initialize result matrix to zeros
    std::memset(result, 0, a_rows * b_cols * sizeof(float));
    
    for (size_t i = 0; i < a_rows; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            float32x4_t sum = vdupq_n_f32(0);
            
            for (size_t k = 0; k < a_cols; k += 4) {
                size_t remaining = std::min(size_t(4), a_cols - k);
                
                if (remaining == 4) {
                    // Load 4 elements from row i of matrix a
                    float32x4_t va = vld1q_f32(&a[i * a_cols + k]);
                    
                    // Load 4 elements from column j of matrix b
                    float temp[4];
                    for (int l = 0; l < 4; l++) {
                        temp[l] = b[(k + l) * b_cols + j];
                    }
                    float32x4_t vb = vld1q_f32(temp);
                    
                    // Multiply and accumulate
                    sum = vmlaq_f32(sum, va, vb);
                } else {
                    // Handle remaining elements
                    for (size_t l = 0; l < remaining; l++) {
                        result[i * b_cols + j] += a[i * a_cols + k + l] * b[(k + l) * b_cols + j];
                    }
                }
            }
            
            // Horizontal sum of the NEON register
            float temp[4];
            vst1q_f32(temp, sum);
            for (int l = 0; l < 4; l++) {
                result[i * b_cols + j] += temp[l];
            }
        }
    }
}
#endif

// Scalar implementation for comparison
void vector_add_scalar(const float* a, const float* b, size_t length, float* result) {
    for (size_t i = 0; i < length; i++) {
        result[i] = a[i] + b[i];
    }
}

void matrix_mul_scalar(const float* a, const float* b, size_t a_rows, size_t a_cols, size_t b_cols, float* result) {
    // Initialize result matrix to zeros
    std::memset(result, 0, a_rows * b_cols * sizeof(float));
    
    for (size_t i = 0; i < a_rows; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a_cols; k++) {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            result[i * b_cols + j] = sum;
        }
    }
}

// Cross-architecture vector addition benchmark
static void BM_CrossArch_VectorAdd(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> a = generate_random_data(size);
    std::vector<float> b = generate_random_data(size);
    std::vector<float> result(size);
    
    std::string arch = state.range(1) == 0 ? "RVV" : 
                      (state.range(1) == 1 ? "AVX" : 
                      (state.range(1) == 2 ? "NEON" : "Scalar"));
    
    for (auto _ : state) {
        if (state.range(1) == 0) {
            // RVV implementation
            rvv_simd::vector_add(a.data(), b.data(), size, result.data());
        }
        else if (state.range(1) == 1) {
            // AVX implementation
            #if defined(__x86_64__) && defined(__AVX__)
            vector_add_avx(a.data(), b.data(), size, result.data());
            #else
            vector_add_scalar(a.data(), b.data(), size, result.data());
            #endif
        }
        else if (state.range(1) == 2) {
            // NEON implementation
            #if defined(__arm__) && defined(__ARM_NEON)
            vector_add_neon(a.data(), b.data(), size, result.data());
            #else
            vector_add_scalar(a.data(), b.data(), size, result.data());
            #endif
        }
        else {
            // Scalar implementation
            vector_add_scalar(a.data(), b.data(), size, result.data());
        }
        
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 3);
    state.SetLabel(arch);
}

// Cross-architecture matrix multiplication benchmark
static void BM_CrossArch_MatrixMul(benchmark::State& state) {
    const size_t size = state.range(0);
    const size_t a_rows = size;
    const size_t a_cols = size;
    const size_t b_cols = size;
    
    std::vector<float> a = generate_random_data(a_rows * a_cols);
    std::vector<float> b = generate_random_data(a_cols * b_cols);
    std::vector<float> result(a_rows * b_cols);
    
    std::string arch = state.range(1) == 0 ? "RVV" : 
                      (state.range(1) == 1 ? "AVX" : 
                      (state.range(1) == 2 ? "NEON" : "Scalar"));
    
    for (auto _ : state) {
        if (state.range(1) == 0) {
            // RVV implementation
            rvv_simd::matrix_mul(a.data(), b.data(), a_rows, a_cols, b_cols, result.data());
        }
        else if (state.range(1) == 1) {
            // AVX implementation
            #if defined(__x86_64__) && defined(__AVX__)
            matrix_mul_avx(a.data(), b.data(), a_rows, a_cols, b_cols, result.data());
            #else
            matrix_mul_scalar(a.data(), b.data(), a_rows, a_cols, b_cols, result.data());
            #endif
        }
        else if (state.range(1) == 2) {
            // NEON implementation
            #if defined(__arm__) && defined(__ARM_NEON)
            matrix_mul_neon(a.data(), b.data(), a_rows, a_cols, b_cols, result.data());
            #else
            matrix_mul_scalar(a.data(), b.data(), a_rows, a_cols, b_cols, result.data());
            #endif
        }
        else {
            // Scalar implementation
            matrix_mul_scalar(a.data(), b.data(), a_rows, a_cols, b_cols, result.data());
        }
        
        benchmark::DoNotOptimize(result.data());
    }
    
    // For matrix multiplication, we perform a_rows * a_cols * b_cols operations
    state.SetItemsProcessed(state.iterations() * a_rows * a_cols * b_cols);
    state.SetBytesProcessed(state.iterations() * (a_rows * a_cols + a_cols * b_cols + a_rows * b_cols) * sizeof(float));
    state.SetLabel(arch);
}

// Register benchmarks with different vector sizes and architectures
// Architecture codes: 0 = RVV, 1 = AVX, 2 = NEON, 3 = Scalar
BENCHMARK(BM_CrossArch_VectorAdd)
    ->Args({1024, 0})    // RVV, 1K elements
    ->Args({1024, 1})    // AVX, 1K elements
    ->Args({1024, 2})    // NEON, 1K elements
    ->Args({1024, 3})    // Scalar, 1K elements
    ->Args({1024*1024, 0})    // RVV, 1M elements
    ->Args({1024*1024, 1})    // AVX, 1M elements
    ->Args({1024*1024, 2})    // NEON, 1M elements
    ->Args({1024*1024, 3});   // Scalar, 1M elements

BENCHMARK(BM_CrossArch_MatrixMul)
    ->Args({64, 0})    // RVV, 64x64 matrix
    ->Args({64, 1})    // AVX, 64x64 matrix
    ->Args({64, 2})    // NEON, 64x64 matrix
    ->Args({64, 3})    // Scalar, 64x64 matrix
    ->Args({256, 0})   // RVV, 256x256 matrix
    ->Args({256, 1})   // AVX, 256x256 matrix
    ->Args({256, 2})   // NEON, 256x256 matrix
    ->Args({256, 3});  // Scalar, 256x256 matrix

// Main function is provided by BENCHMARK_MAIN() in the other file
