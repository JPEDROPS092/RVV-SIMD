#include <gtest/gtest.h>
#include "rvv_simd.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class VectorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        
        // Generate random vectors
        a.resize(size);
        b.resize(size);
        result.resize(size);
        
        for (size_t i = 0; i < size; i++) {
            a[i] = dist(gen);
            b[i] = dist(gen);
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
    
    // Vector size for tests
    const size_t size = 1024;
    
    // Test vectors
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> result;
};

// Test vector addition
TEST_F(VectorOpsTest, VectorAdd) {
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] + b[i];
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_add(a.data(), b.data(), size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test vector subtraction
TEST_F(VectorOpsTest, VectorSub) {
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] - b[i];
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_sub(a.data(), b.data(), size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test vector multiplication
TEST_F(VectorOpsTest, VectorMul) {
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] * b[i];
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_mul(a.data(), b.data(), size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test vector division
TEST_F(VectorOpsTest, VectorDiv) {
    // Avoid division by zero
    for (size_t i = 0; i < size; i++) {
        if (std::abs(b[i]) < 1e-5f) {
            b[i] = 1.0f;
        }
    }
    
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] / b[i];
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_div(a.data(), b.data(), size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test vector dot product
TEST_F(VectorOpsTest, VectorDot) {
    // Compute expected result
    float expected = 0.0f;
    for (size_t i = 0; i < size; i++) {
        expected += a[i] * b[i];
    }
    
    // Compute result using RVV-SIMD
    float result = rvv_simd::vector_dot(a.data(), b.data(), size);
    
    // Check if results match
    EXPECT_NEAR(result, expected, 1e-3f);
}

// Test vector scaling
TEST_F(VectorOpsTest, VectorScale) {
    float scalar = 2.5f;
    
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] * scalar;
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_scale(a.data(), scalar, size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test vector norm
TEST_F(VectorOpsTest, VectorNorm) {
    // Compute expected result
    float expected = 0.0f;
    for (size_t i = 0; i < size; i++) {
        expected += a[i] * a[i];
    }
    expected = std::sqrt(expected);
    
    // Compute result using RVV-SIMD
    float result = rvv_simd::vector_norm(a.data(), size);
    
    // Check if results match
    EXPECT_NEAR(result, expected, 1e-3f);
}

// Test vector normalization
TEST_F(VectorOpsTest, VectorNormalize) {
    // Compute the norm
    float norm = 0.0f;
    for (size_t i = 0; i < size; i++) {
        norm += a[i] * a[i];
    }
    norm = std::sqrt(norm);
    
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] / norm;
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_normalize(a.data(), size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
    
    // Check if the result has unit norm
    float result_norm = rvv_simd::vector_norm(result.data(), size);
    EXPECT_NEAR(result_norm, 1.0f, 1e-5f);
}

// Test vector sigmoid
TEST_F(VectorOpsTest, VectorSigmoid) {
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_sigmoid(a.data(), size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected, 1e-4f));
}

// Test vector ReLU
TEST_F(VectorOpsTest, VectorReLU) {
    // Compute expected result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = std::max(0.0f, a[i]);
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::vector_relu(a.data(), size, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}
