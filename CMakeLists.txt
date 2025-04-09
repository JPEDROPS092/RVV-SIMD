cmake_minimum_required(VERSION 3.10)
project(rvv_simd VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_RVV "Use RISC-V Vector extensions if available" ON)

# Check if we're on RISC-V
execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE MACHINE_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set compiler flags
if(MACHINE_ARCH MATCHES "riscv")
    message(STATUS "Building for RISC-V architecture: ${MACHINE_ARCH}")
    if(USE_RVV)
        # Add RVV-specific flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv -mabi=lp64d")
        add_definitions(-D__riscv_vector)
        message(STATUS "RISC-V Vector extensions enabled")
    endif()
else()
    message(STATUS "Building for non-RISC-V architecture: ${MACHINE_ARCH}")
    # Add architecture-specific optimizations for x86/ARM
    if(MACHINE_ARCH MATCHES "x86_64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        add_definitions(-D__x86_64__)
    elseif(MACHINE_ARCH MATCHES "arm")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
        add_definitions(-D__arm__)
    endif()
endif()

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE SOURCES "src/core/*.cpp" "src/ml/*.cpp" "src/utils/*.cpp")

# Create library
add_library(rvv_simd ${SOURCES})
set_target_properties(rvv_simd PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Install library and headers
install(TARGETS rvv_simd
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# Build tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Build Python bindings
if(BUILD_PYTHON_BINDINGS)
    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Find pybind11
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.10.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    # Create Python module
    pybind11_add_module(rvv_simd_python python/rvv_simd_bindings.cpp)
    target_link_libraries(rvv_simd_python PRIVATE rvv_simd)
    set_target_properties(rvv_simd_python PROPERTIES
        OUTPUT_NAME rvv_simd
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/rvv_simd
    )
endif()

# Add examples
add_subdirectory(examples)
