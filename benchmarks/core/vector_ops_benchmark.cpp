#include <benchmark/benchmark.h>
#include "rvv_simd.h"
#include <vector>
#include <random>
#include <cmath>

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

// Benchmark vector addition
static void BM_VectorAdd(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> a = generate_random_data(size);
    std::vector<float> b = generate_random_data(size);
    std::vector<float> result(size);
    
    for (auto _ : state) {
        rvv_simd::vector_add(a.data(), b.data(), size, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 3); // 2 inputs + 1 output
}

// Benchmark vector multiplication
static void BM_VectorMul(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> a = generate_random_data(size);
    std::vector<float> b = generate_random_data(size);
    std::vector<float> result(size);
    
    for (auto _ : state) {
        rvv_simd::vector_mul(a.data(), b.data(), size, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 3);
}

// Benchmark dot product
static void BM_VectorDot(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> a = generate_random_data(size);
    std::vector<float> b = generate_random_data(size);
    float result;
    
    for (auto _ : state) {
        result = rvv_simd::vector_dot(a.data(), b.data(), size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
}

// Benchmark vector normalization
static void BM_VectorNormalize(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> a = generate_random_data(size);
    std::vector<float> result(size);
    
    for (auto _ : state) {
        rvv_simd::vector_normalize(a.data(), size, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
}

// Benchmark sigmoid function
static void BM_VectorSigmoid(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> a = generate_random_data(size);
    std::vector<float> result(size);
    
    for (auto _ : state) {
        rvv_simd::vector_sigmoid(a.data(), size, result.data());
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
}

// Register benchmarks with different vector sizes
BENCHMARK(BM_VectorAdd)->RangeMultiplier(4)->Range(1024, 1024*1024);
BENCHMARK(BM_VectorMul)->RangeMultiplier(4)->Range(1024, 1024*1024);
BENCHMARK(BM_VectorDot)->RangeMultiplier(4)->Range(1024, 1024*1024);
BENCHMARK(BM_VectorNormalize)->RangeMultiplier(4)->Range(1024, 1024*1024);
BENCHMARK(BM_VectorSigmoid)->RangeMultiplier(4)->Range(1024, 1024*1024);

// Main function
BENCHMARK_MAIN();
