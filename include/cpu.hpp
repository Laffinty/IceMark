#pragma once
#include <cstdint>
#include <string>

namespace icemark {

struct CPUInfo {
    uint32_t logical_cores = 0;
    uint32_t physical_cores = 0;
    uint32_t max_freq_mhz = 0;   // nominal, from CPUID / registry
    bool has_ht = false;
    bool has_fma = false;
    bool has_avx2 = false;
    bool has_avx512 = false;     // ISA is supported AND the OS enabled ZMM state
    bool has_amx = false;
    bool has_neon = false;
    std::string brand;
};

CPUInfo detect_cpu();

std::string simd_capabilities_string(const CPUInfo& info);

// Physical RAM and currently-available RAM. The benchmark must size itself
// against *available*, never against total, or it will happily over-commit.
struct MemoryInfo {
    uint64_t total_bytes = 0;
    uint64_t avail_bytes = 0;
};
MemoryInfo detect_memory();

} // namespace icemark