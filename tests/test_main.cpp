#include <gtest/gtest.h>
#include "rvv_simd.h"

int main(int argc, char **argv) {
    // Initialize the RVV-SIMD library
    rvv_simd::initialize();
    
    // Print RVV information
    std::cout << "RVV-SIMD Version: " << rvv_simd::get_version() << std::endl;
    std::cout << "RVV Support: " << (rvv_simd::is_rvv_supported() ? "Yes" : "No") << std::endl;
    std::cout << "RVV Info: " << rvv_simd::get_rvv_info() << std::endl;
    
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Run all tests
    return RUN_ALL_TESTS();
}
