cmake_minimum_required(VERSION 3.10)

# Find GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add test executable
add_executable(rvv_simd_tests
  test_main.cpp
  test_vector_ops.cpp
  test_matrix_ops.cpp
  test_ml_ops.cpp
)

# Link against GTest and the RVV-SIMD library
target_link_libraries(rvv_simd_tests
  PRIVATE
  rvv_simd
  gtest
  gtest_main
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(rvv_simd_tests)
