cmake_minimum_required(VERSION 3.10)

# Find Google Benchmark
include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.7.0
)
FetchContent_MakeAvailable(benchmark)

# Collect benchmark source files
file(GLOB_RECURSE BENCHMARK_SOURCES "core/*.cpp" "ml/*.cpp" "cross_arch/*.cpp")

# Create benchmark executable
add_executable(rvv_simd_benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(rvv_simd_benchmarks PRIVATE rvv_simd benchmark::benchmark)

# Install benchmarks
install(TARGETS rvv_simd_benchmarks DESTINATION bin)
