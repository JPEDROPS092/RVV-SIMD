#include <benchmark/benchmark.h>
#include "rvv_simd.h"
#include <vector>
#include <random>
#include <cmath>

// Helper function to generate random matrix data
std::vector<float> generate_random_matrix(size_t rows, size_t cols) {
    std::vector<float> data(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = dist(gen);
    }
    
    return data;
}

// Benchmark matrix addition
static void BM_MatrixAdd(benchmark::State& state) {
    const size_t size = state.range(0);
    const size_t rows = size;
    const size_t cols = size;
    
    std::vector<float> a = generate_random_matrix(rows, cols);
    std::vector<float> b = generate_random_matrix(rows, cols);
    std::vector<float> result(rows * cols);
    
    for (auto _ : state) {
        rvv_simd::matrix_add(a.data(), b.data(), rows, cols, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * rows * cols);
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(float) * 3); // 2 inputs + 1 output
}

// Benchmark matrix multiplication
static void BM_MatrixMul(benchmark::State& state) {
    const size_t size = state.range(0);
    const size_t a_rows = size;
    const size_t a_cols = size;
    const size_t b_cols = size;
    
    std::vector<float> a = generate_random_matrix(a_rows, a_cols);
    std::vector<float> b = generate_random_matrix(a_cols, b_cols);
    std::vector<float> result(a_rows * b_cols);
    
    for (auto _ : state) {
        rvv_simd::matrix_mul(a.data(), b.data(), a_rows, a_cols, b_cols, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    // For matrix multiplication, we perform a_rows * a_cols * b_cols operations
    state.SetItemsProcessed(state.iterations() * a_rows * a_cols * b_cols);
    state.SetBytesProcessed(state.iterations() * (a_rows * a_cols + a_cols * b_cols + a_rows * b_cols) * sizeof(float));
}

// Benchmark matrix transpose
static void BM_MatrixTranspose(benchmark::State& state) {
    const size_t size = state.range(0);
    const size_t rows = size;
    const size_t cols = size;
    
    std::vector<float> a = generate_random_matrix(rows, cols);
    std::vector<float> result(cols * rows);
    
    for (auto _ : state) {
        rvv_simd::matrix_transpose(a.data(), rows, cols, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * rows * cols);
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(float) * 2); // 1 input + 1 output
}

// Benchmark matrix scaling
static void BM_MatrixScale(benchmark::State& state) {
    const size_t size = state.range(0);
    const size_t rows = size;
    const size_t cols = size;
    
    std::vector<float> a = generate_random_matrix(rows, cols);
    std::vector<float> result(rows * cols);
    const float scalar = 2.5f;
    
    for (auto _ : state) {
        rvv_simd::matrix_scale(a.data(), scalar, rows, cols, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * rows * cols);
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(float) * 2); // 1 input + 1 output
}

// Register benchmarks with different matrix sizes
BENCHMARK(BM_MatrixAdd)->RangeMultiplier(2)->Range(32, 1024);
BENCHMARK(BM_MatrixMul)->RangeMultiplier(2)->Range(32, 512);  // Matrix multiplication is O(n³), so we use smaller sizes
BENCHMARK(BM_MatrixTranspose)->RangeMultiplier(2)->Range(32, 1024);
BENCHMARK(BM_MatrixScale)->RangeMultiplier(2)->Range(32, 1024);

// Main function is provided by BENCHMARK_MAIN() in the other file
