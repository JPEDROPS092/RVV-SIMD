#include <benchmark/benchmark.h>
#include "rvv_simd.h"
#include <vector>
#include <random>
#include <cmath>

// Helper function to generate random tensor data
std::vector<float> generate_random_tensor(size_t size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
    
    return data;
}

// Benchmark convolution operation
static void BM_Convolution2D(benchmark::State& state) {
    const size_t input_size = state.range(0);
    const size_t kernel_size = 3;  // Fixed 3x3 kernel
    const size_t input_channels = 3;  // RGB
    const size_t output_channels = 16;  // Number of filters
    const size_t stride = 1;
    const size_t padding = 1;
    
    // Create input tensor (C x H x W)
    std::vector<float> input = generate_random_tensor(input_channels * input_size * input_size);
    
    // Create kernel tensor (N x C x H x W)
    std::vector<float> kernel = generate_random_tensor(output_channels * input_channels * kernel_size * kernel_size);
    
    // Calculate output size
    const size_t output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
    std::vector<float> output(output_channels * output_size * output_size);
    
    for (auto _ : state) {
        rvv_simd::convolution_2d(
            input.data(), kernel.data(),
            input_size, input_size, input_channels,
            kernel_size, kernel_size, input_channels, output_channels,
            stride, stride,
            padding, padding,
            output.data()
        );
        benchmark::DoNotOptimize(output.data());
    }
    
    // Each output element requires kernel_size * kernel_size * input_channels multiplications and additions
    const size_t ops_per_output = 2 * kernel_size * kernel_size * input_channels;
    const size_t total_ops = output_channels * output_size * output_size * ops_per_output;
    
    state.SetItemsProcessed(state.iterations() * total_ops);
    state.SetBytesProcessed(state.iterations() * (
        input_channels * input_size * input_size + 
        output_channels * input_channels * kernel_size * kernel_size + 
        output_channels * output_size * output_size
    ) * sizeof(float));
}

// Benchmark max pooling operation
static void BM_MaxPooling2D(benchmark::State& state) {
    const size_t input_size = state.range(0);
    const size_t pool_size = 2;
    const size_t stride = 2;
    const size_t channels = 16;
    
    // Create input tensor (C x H x W)
    std::vector<float> input = generate_random_tensor(channels * input_size * input_size);
    
    // Calculate output size
    const size_t output_size = (input_size - pool_size) / stride + 1;
    std::vector<float> output(channels * output_size * output_size);
    
    for (auto _ : state) {
        rvv_simd::max_pooling_2d(
            input.data(),
            input_size, input_size, channels,
            pool_size, pool_size,
            stride, stride,
            output.data()
        );
        benchmark::DoNotOptimize(output.data());
    }
    
    // Each output element requires pool_size * pool_size comparisons
    const size_t ops_per_output = pool_size * pool_size;
    const size_t total_ops = channels * output_size * output_size * ops_per_output;
    
    state.SetItemsProcessed(state.iterations() * total_ops);
    state.SetBytesProcessed(state.iterations() * (
        channels * input_size * input_size + 
        channels * output_size * output_size
    ) * sizeof(float));
}

// Benchmark batch normalization
static void BM_BatchNorm(benchmark::State& state) {
    const size_t batch_size = state.range(0);
    const size_t channels = 64;
    const size_t height = 32;
    const size_t width = 32;
    const size_t feature_size = height * width;
    const float epsilon = 1e-5f;
    
    // Create input tensor (N x C x H x W)
    std::vector<float> input = generate_random_tensor(batch_size * channels * feature_size);
    
    // Create parameters
    std::vector<float> gamma = generate_random_tensor(channels);
    std::vector<float> beta = generate_random_tensor(channels);
    std::vector<float> mean = generate_random_tensor(channels);
    std::vector<float> var = generate_random_tensor(channels);
    
    // Make sure variances are positive
    for (size_t i = 0; i < channels; i++) {
        var[i] = std::abs(var[i]) + 0.1f;
    }
    
    // Output tensor
    std::vector<float> output(batch_size * channels * feature_size);
    
    for (auto _ : state) {
        rvv_simd::batch_norm(
            input.data(), gamma.data(), beta.data(),
            mean.data(), var.data(), epsilon,
            feature_size * batch_size, channels,
            output.data()
        );
        benchmark::DoNotOptimize(output.data());
    }
    
    // Each element requires 5 operations (subtract, divide, multiply, add)
    const size_t ops_per_element = 5;
    const size_t total_elements = batch_size * channels * feature_size;
    
    state.SetItemsProcessed(state.iterations() * total_elements * ops_per_element);
    state.SetBytesProcessed(state.iterations() * (
        total_elements + // input
        channels * 4 +   // gamma, beta, mean, var
        total_elements   // output
    ) * sizeof(float));
}

// Benchmark softmax
static void BM_Softmax(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // Create input vector
    std::vector<float> input = generate_random_tensor(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        rvv_simd::softmax(input.data(), size, output.data());
        benchmark::DoNotOptimize(output.data());
    }
    
    // Softmax requires finding max, computing exp for each element, summing, and normalizing
    const size_t ops_per_element = 4;
    
    state.SetItemsProcessed(state.iterations() * size * ops_per_element);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
}

// Register benchmarks with different sizes
BENCHMARK(BM_Convolution2D)->RangeMultiplier(2)->Range(16, 256);
BENCHMARK(BM_MaxPooling2D)->RangeMultiplier(2)->Range(16, 512);
BENCHMARK(BM_BatchNorm)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(BM_Softmax)->RangeMultiplier(2)->Range(128, 16384);

// Main function is provided by BENCHMARK_MAIN() in the other file
