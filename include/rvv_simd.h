/**
 * @file rvv_simd.h
 * @brief Main header file for the RVV-SIMD library
 * 
 * This header includes all the necessary components of the RVV-SIMD library,
 * providing a single include point for applications.
 */

#ifndef RVV_SIMD_H
#define RVV_SIMD_H

#include "rvv_simd/vector_ops.h"
#include "rvv_simd/matrix_ops.h"
#include "rvv_simd/ml_ops.h"
#include "rvv_simd/config.h"

/**
 * @namespace rvv_simd
 * @brief Namespace containing all RVV-SIMD library functions and classes
 */
namespace rvv_simd {

/**
 * @brief Initialize the RVV-SIMD library
 * 
 * This function should be called before using any other functions in the library.
 * It performs necessary setup and checks for RVV support.
 * 
 * @return true if initialization was successful, false otherwise
 */
bool initialize();

/**
 * @brief Check if the current hardware supports RVV
 * 
 * @return true if RVV is supported, false otherwise
 */
bool is_rvv_supported();

/**
 * @brief Get the version of the RVV-SIMD library
 * 
 * @return String containing the version number
 */
const char* get_version();

/**
 * @brief Get information about the RVV implementation
 * 
 * @return String containing information about the RVV implementation
 */
const char* get_rvv_info();

} // namespace rvv_simd

#endif // RVV_SIMD_H
