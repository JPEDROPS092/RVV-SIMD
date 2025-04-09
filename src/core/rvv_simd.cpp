#include "rvv_simd.h"
#include "rvv_simd/config.h"
#include <cstring>
#include <string>

// Check if we can use RVV intrinsics
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rvv_simd {

// Global configuration
static Config g_config = get_default_config();

// Version string
static const char* VERSION_STR = "RVV-SIMD v0.1.0";

bool initialize() {
    // Check if RVV is supported
    bool rvv_supported = is_rvv_supported();
    
    if (rvv_supported) {
        // If RVV is supported, detect vector length
        if (g_config.vector_length == 0) {
#if defined(__riscv_vector)
            // Auto-detect vector length
            g_config.vector_length = __riscv_vlenb * 8; // in bits
#else
            g_config.vector_length = 0;
#endif
        }
    } else {
        // If RVV is not supported, disable intrinsics
        g_config.use_intrinsics = false;
        g_config.vector_length = 0;
    }
    
    return true;
}

bool is_rvv_supported() {
#if defined(__riscv_vector)
    return true;
#else
    return false;
#endif
}

const char* get_version() {
    return VERSION_STR;
}

const char* get_rvv_info() {
    static std::string info;
    
    if (info.empty()) {
        info = "RVV-SIMD Information:\n";
        
        if (is_rvv_supported()) {
            info += "- RISC-V Vector (RVV) extension: Supported\n";
#if defined(__riscv_vector)
            info += "- Vector register length (VLEN): " + std::to_string(__riscv_vlenb * 8) + " bits\n";
#endif
        } else {
            info += "- RISC-V Vector (RVV) extension: Not supported\n";
            info += "- Using fallback scalar implementations\n";
        }
        
        info += "- Optimization level: ";
        switch (g_config.opt_level) {
            case OptimizationLevel::NONE:
                info += "None\n";
                break;
            case OptimizationLevel::BASIC:
                info += "Basic\n";
                break;
            case OptimizationLevel::ADVANCED:
                info += "Advanced\n";
                break;
            default:
                info += "Unknown\n";
                break;
        }
        
        info += "- Using intrinsics: " + std::string(g_config.use_intrinsics ? "Yes" : "No") + "\n";
    }
    
    return info.c_str();
}

Config get_default_config() {
    Config config;
    config.opt_level = OptimizationLevel::ADVANCED;
    config.use_intrinsics = true;
    config.enable_profiling = false;
    config.enable_logging = false;
    config.vector_length = 0;  // Auto-detect
    
    return config;
}

void set_config(const Config& config) {
    g_config = config;
}

Config get_config() {
    return g_config;
}

} // namespace rvv_simd
