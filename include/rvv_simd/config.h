/**
 * @file config.h
 * @brief Configuration settings for the RVV-SIMD library
 * 
 * This file contains configuration constants and settings for the RVV-SIMD library.
 */

#ifndef RVV_SIMD_CONFIG_H
#define RVV_SIMD_CONFIG_H

namespace rvv_simd {

/**
 * @brief Library version information
 */
#define RVV_SIMD_VERSION_MAJOR 0
#define RVV_SIMD_VERSION_MINOR 1
#define RVV_SIMD_VERSION_PATCH 0

/**
 * @brief Configuration options for RVV-SIMD
 */
enum class OptimizationLevel {
    NONE = 0,       ///< No optimization
    BASIC = 1,      ///< Basic RVV optimization
    ADVANCED = 2    ///< Advanced RVV optimization with auto-vectorization
};

/**
 * @brief Data types supported by the library
 */
enum class DataType {
    FLOAT32,        ///< 32-bit floating point
    FLOAT64,        ///< 64-bit floating point
    INT8,           ///< 8-bit integer
    INT16,          ///< 16-bit integer
    INT32,          ///< 32-bit integer
    INT64           ///< 64-bit integer
};

/**
 * @brief Configuration structure for the library
 */
struct Config {
    OptimizationLevel opt_level = OptimizationLevel::ADVANCED;
    bool use_intrinsics = true;
    bool enable_profiling = false;
    bool enable_logging = false;
    int vector_length = 0;  // 0 means auto-detect
};

/**
 * @brief Get the default configuration
 * 
 * @return Default configuration
 */
Config get_default_config();

/**
 * @brief Set the global configuration
 * 
 * @param config Configuration to set
 */
void set_config(const Config& config);

/**
 * @brief Get the current global configuration
 * 
 * @return Current configuration
 */
Config get_config();

} // namespace rvv_simd

#endif // RVV_SIMD_CONFIG_H
