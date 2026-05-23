#pragma once
#include <cstdint>
#include <string>

struct CPUInfo {
    uint32_t logical_cores;
    uint32_t physical_cores;
    bool has_ht;
    bool has_avx2;
    bool has_avx512;
    bool has_amx;
    bool has_neon;
};

CPUInfo detect_cpu();

std::string simd_capabilities_string(const CPUInfo& info);