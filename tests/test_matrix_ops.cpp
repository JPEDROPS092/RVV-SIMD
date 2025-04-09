#include <gtest/gtest.h>
#include "rvv_simd.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class MatrixOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        
        // Generate random matrices
        a.resize(rows * cols);
        b.resize(rows * cols);
        result.resize(rows * cols);
        
        for (size_t i = 0; i < rows * cols; i++) {
            a[i] = dist(gen);
            b[i] = dist(gen);
        }
    }
    
    // Helper function to check if two matrices are approximately equal
    bool approx_equal(const std::vector<float>& x, const std::vector<float>& y, float epsilon = 1e-5f) {
        if (x.size() != y.size()) return false;
        
        for (size_t i = 0; i < x.size(); i++) {
            if (std::abs(x[i] - y[i]) > epsilon) return false;
        }
        
        return true;
    }
    
    // Matrix dimensions for tests
    const size_t rows = 32;
    const size_t cols = 32;
    
    // Test matrices
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> result;
};

// Test matrix addition
TEST_F(MatrixOpsTest, MatrixAdd) {
    // Compute expected result
    std::vector<float> expected(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        expected[i] = a[i] + b[i];
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::matrix_add(a.data(), b.data(), rows, cols, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test matrix subtraction
TEST_F(MatrixOpsTest, MatrixSub) {
    // Compute expected result
    std::vector<float> expected(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        expected[i] = a[i] - b[i];
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::matrix_sub(a.data(), b.data(), rows, cols, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test element-wise matrix multiplication
TEST_F(MatrixOpsTest, MatrixElemMul) {
    // Compute expected result
    std::vector<float> expected(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        expected[i] = a[i] * b[i];
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::matrix_elem_mul(a.data(), b.data(), rows, cols, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test matrix multiplication
TEST_F(MatrixOpsTest, MatrixMul) {
    // Create matrices for multiplication
    const size_t a_rows = 16;
    const size_t a_cols = 24;
    const size_t b_cols = 32;
    
    std::vector<float> a_mat(a_rows * a_cols);
    std::vector<float> b_mat(a_cols * b_cols);
    std::vector<float> result_mat(a_rows * b_cols);
    
    // Fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    for (size_t i = 0; i < a_rows * a_cols; i++) {
        a_mat[i] = dist(gen);
    }
    
    for (size_t i = 0; i < a_cols * b_cols; i++) {
        b_mat[i] = dist(gen);
    }
    
    // Compute expected result
    std::vector<float> expected(a_rows * b_cols, 0.0f);
    for (size_t i = 0; i < a_rows; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a_cols; k++) {
                sum += a_mat[i * a_cols + k] * b_mat[k * b_cols + j];
            }
            expected[i * b_cols + j] = sum;
        }
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::matrix_mul(a_mat.data(), b_mat.data(), a_rows, a_cols, b_cols, result_mat.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result_mat, expected, 1e-3f));
}

// Test matrix transpose
TEST_F(MatrixOpsTest, MatrixTranspose) {
    // Compute expected result
    std::vector<float> expected(cols * rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            expected[j * rows + i] = a[i * cols + j];
        }
    }
    
    // Compute result using RVV-SIMD
    std::vector<float> result_t(cols * rows);
    rvv_simd::matrix_transpose(a.data(), rows, cols, result_t.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result_t, expected));
}

// Test matrix scaling
TEST_F(MatrixOpsTest, MatrixScale) {
    float scalar = 2.5f;
    
    // Compute expected result
    std::vector<float> expected(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        expected[i] = a[i] * scalar;
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::matrix_scale(a.data(), scalar, rows, cols, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}

// Test matrix sum
TEST_F(MatrixOpsTest, MatrixSum) {
    // Compute expected result
    float expected = 0.0f;
    for (size_t i = 0; i < rows * cols; i++) {
        expected += a[i];
    }
    
    // Compute result using RVV-SIMD
    float result = rvv_simd::matrix_sum(a.data(), rows, cols);
    
    // Check if results match
    EXPECT_NEAR(result, expected, 1e-3f);
}

// Test matrix norm
TEST_F(MatrixOpsTest, MatrixNorm) {
    // Compute expected result
    float expected = 0.0f;
    for (size_t i = 0; i < rows * cols; i++) {
        expected += a[i] * a[i];
    }
    expected = std::sqrt(expected);
    
    // Compute result using RVV-SIMD
    float result = rvv_simd::matrix_norm(a.data(), rows, cols);
    
    // Check if results match
    EXPECT_NEAR(result, expected, 1e-3f);
}

// Test matrix apply function
TEST_F(MatrixOpsTest, MatrixApply) {
    // Define a simple function to apply
    auto square = [](float x) -> float { return x * x; };
    
    // Compute expected result
    std::vector<float> expected(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        expected[i] = square(a[i]);
    }
    
    // Compute result using RVV-SIMD
    rvv_simd::matrix_apply(a.data(), rows, cols, square, result.data());
    
    // Check if results match
    EXPECT_TRUE(approx_equal(result, expected));
}
