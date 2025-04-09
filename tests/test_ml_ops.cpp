#include <gtest/gtest.h>
#include "rvv_simd.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class MLOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Generate random input tensor
        input.resize(input_c * input_h * input_w);
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = dist(gen);
        }
        
        // Generate random kernel tensor
        kernel.resize(kernel_n * kernel_c * kernel_h * kernel_w);
        for (size_t i = 0; i < kernel.size(); i++) {
            kernel[i] = dist(gen);
        }
    }
    
    // Helper function to check if two vectors are approximately equal
    bool approx_equal(const std::vector<float>& x, const std::vector<float>& y, float epsilon = 1e-5f) {
        if (x.size() != y.size()) return false;
        
        for (size_t i = 0; i < x.size(); i++) {
            if (std::abs(x[i] - y[i]) > epsilon) return false;
        }
        
        return true;
    }
    
    // Input tensor dimensions
    const size_t input_c = 3;  // channels
    const size_t input_h = 16; // height
    const size_t input_w = 16; // width
    
    // Kernel dimensions
    const size_t kernel_n = 8;  // number of kernels
    const size_t kernel_c = 3;  // channels (must match input_c)
    const size_t kernel_h = 3;  // height
    const size_t kernel_w = 3;  // width
    
    // Convolution parameters
    const size_t stride_h = 1;
    const size_t stride_w = 1;
    const size_t padding_h = 1;
    const size_t padding_w = 1;
    
    // Test data
    std::vector<float> input;
    std::vector<float> kernel;
};

// Test convolution operation
TEST_F(MLOpsTest, Convolution2D) {
    // Calculate output dimensions
    const size_t output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
    const size_t output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    // Allocate output tensor
    std::vector<float> output(kernel_n * output_h * output_w);
    
    // Compute result using RVV-SIMD
    rvv_simd::convolution_2d(
        input.data(), kernel.data(),
        input_h, input_w, input_c,
        kernel_h, kernel_w, kernel_c, kernel_n,
        stride_h, stride_w,
        padding_h, padding_w,
        output.data()
    );
    
    // Compute expected result using naive implementation
    std::vector<float> expected(kernel_n * output_h * output_w, 0.0f);
    
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
                
                expected[(n * output_h + oh) * output_w + ow] = sum;
            }
        }
    }
    
    // Check if results match
    EXPECT_TRUE(approx_equal(output, expected, 1e-4f));
}

// Test max pooling operation
TEST_F(MLOpsTest, MaxPooling2D) {
    // Pooling parameters
    const size_t pool_h = 2;
    const size_t pool_w = 2;
    const size_t stride_h = 2;
    const size_t stride_w = 2;
    
    // Calculate output dimensions
    const size_t output_h = (input_h - pool_h) / stride_h + 1;
    const size_t output_w = (input_w - pool_w) / stride_w + 1;
    
    // Allocate output tensor
    std::vector<float> output(input_c * output_h * output_w);
    
    // Compute result using RVV-SIMD
    rvv_simd::max_pooling_2d(
        input.data(),
        input_h, input_w, input_c,
        pool_h, pool_w,
        stride_h, stride_w,
        output.data()
    );
    
    // Compute expected result using naive implementation
    std::vector<float> expected(input_c * output_h * output_w);
    
    for (size_t c = 0; c < input_c; c++) {
        for (size_t oh = 0; oh < output_h; oh++) {
            for (size_t ow = 0; ow < output_w; ow++) {
                float max_val = -std::numeric_limits<float>::infinity();
                
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
                
                expected[(c * output_h + oh) * output_w + ow] = max_val;
            }
        }
    }
    
    // Check if results match
    EXPECT_TRUE(approx_equal(output, expected));
}

// Test average pooling operation
TEST_F(MLOpsTest, AvgPooling2D) {
    // Pooling parameters
    const size_t pool_h = 2;
    const size_t pool_w = 2;
    const size_t stride_h = 2;
    const size_t stride_w = 2;
    
    // Calculate output dimensions
    const size_t output_h = (input_h - pool_h) / stride_h + 1;
    const size_t output_w = (input_w - pool_w) / stride_w + 1;
    
    // Allocate output tensor
    std::vector<float> output(input_c * output_h * output_w);
    
    // Compute result using RVV-SIMD
    rvv_simd::avg_pooling_2d(
        input.data(),
        input_h, input_w, input_c,
        pool_h, pool_w,
        stride_h, stride_w,
        output.data()
    );
    
    // Compute expected result using naive implementation
    std::vector<float> expected(input_c * output_h * output_w);
    
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
                
                expected[(c * output_h + oh) * output_w + ow] = (count > 0) ? (sum / count) : 0.0f;
            }
        }
    }
    
    // Check if results match
    EXPECT_TRUE(approx_equal(output, expected));
}

// Test batch normalization
TEST_F(MLOpsTest, BatchNorm) {
    // Batch normalization parameters
    const size_t channels = 4;
    const size_t height = 8;
    const size_t width = 8;
    const size_t size = height * width;
    const float epsilon = 1e-5f;
    
    // Create input tensor
    std::vector<float> input(channels * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = dist(gen);
    }
    
    // Create parameters
    std::vector<float> gamma(channels);
    std::vector<float> beta(channels);
    std::vector<float> mean(channels);
    std::vector<float> var(channels);
    
    for (size_t c = 0; c < channels; c++) {
        gamma[c] = dist(gen);
        beta[c] = dist(gen);
        mean[c] = dist(gen);
        var[c] = std::abs(dist(gen)) + 0.1f;  // Ensure positive variance
    }
    
    // Allocate output tensor
    std::vector<float> output(channels * size);
    
    // Compute result using RVV-SIMD
    rvv_simd::batch_norm(
        input.data(), gamma.data(), beta.data(),
        mean.data(), var.data(), epsilon,
        size, channels,
        output.data()
    );
    
    // Compute expected result using naive implementation
    std::vector<float> expected(channels * size);
    
    for (size_t c = 0; c < channels; c++) {
        float scale = gamma[c] / std::sqrt(var[c] + epsilon);
        float shift = beta[c] - scale * mean[c];
        
        for (size_t i = 0; i < size; i++) {
            expected[c * size + i] = scale * input[c * size + i] + shift;
        }
    }
    
    // Check if results match
    EXPECT_TRUE(approx_equal(output, expected, 1e-4f));
}

// Test softmax function
TEST_F(MLOpsTest, Softmax) {
    // Create input vector
    const size_t size = 100;
    std::vector<float> input(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (size_t i = 0; i < size; i++) {
        input[i] = dist(gen);
    }
    
    // Allocate output vector
    std::vector<float> output(size);
    
    // Compute result using RVV-SIMD
    rvv_simd::softmax(input.data(), size, output.data());
    
    // Compute expected result using naive implementation
    std::vector<float> expected(size);
    
    // Find max value for numerical stability
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exp(x - max) for each element
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        expected[i] = std::exp(input[i] - max_val);
        sum += expected[i];
    }
    
    // Normalize by sum
    for (size_t i = 0; i < size; i++) {
        expected[i] /= sum;
    }
    
    // Check if results match
    EXPECT_TRUE(approx_equal(output, expected, 1e-5f));
    
    // Check if the sum of probabilities is 1
    float output_sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output_sum += output[i];
    }
    EXPECT_NEAR(output_sum, 1.0f, 1e-5f);
}

// Test cross-entropy loss
TEST_F(MLOpsTest, CrossEntropyLoss) {
    // Create input tensors
    const size_t batch_size = 8;
    const size_t num_classes = 10;
    
    std::vector<float> predictions(batch_size * num_classes);
    std::vector<float> targets(batch_size * num_classes, 0.0f);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.01f, 0.99f);
    
    // Generate random predictions and one-hot targets
    for (size_t b = 0; b < batch_size; b++) {
        // Generate random probabilities and normalize
        float sum = 0.0f;
        for (size_t c = 0; c < num_classes; c++) {
            predictions[b * num_classes + c] = dist(gen);
            sum += predictions[b * num_classes + c];
        }
        for (size_t c = 0; c < num_classes; c++) {
            predictions[b * num_classes + c] /= sum;
        }
        
        // Set one-hot target
        size_t target_class = gen() % num_classes;
        targets[b * num_classes + target_class] = 1.0f;
    }
    
    // Compute result using RVV-SIMD
    float loss = rvv_simd::cross_entropy_loss(
        predictions.data(), targets.data(),
        batch_size, num_classes
    );
    
    // Compute expected result using naive implementation
    float expected_loss = 0.0f;
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < num_classes; c++) {
            if (targets[b * num_classes + c] > 0.0f) {
                // Add a small epsilon to avoid log(0)
                float pred = std::max(predictions[b * num_classes + c], 1e-7f);
                expected_loss -= targets[b * num_classes + c] * std::log(pred);
            }
        }
    }
    
    expected_loss /= batch_size;
    
    // Check if results match
    EXPECT_NEAR(loss, expected_loss, 1e-4f);
}
