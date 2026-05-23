#include "cpu_detector.hpp"
#include <vector>
#include <string>
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <cpuid.h>
#endif

static void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#if defined(_WIN32) || defined(_WIN64)
    int info[4] = {0};
    __cpuidex(info, leaf, subleaf);
    *eax = info[0]; *ebx = info[1]; *ecx = info[2]; *edx = info[3];
#elif defined(__linux__) || defined(__APPLE__)
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}

CPUInfo detect_cpu() {
    CPUInfo info = {0};
    info.logical_cores = std::thread::hardware_concurrency();
    if (info.logical_cores == 0) info.logical_cores = 4;
    info.physical_cores = info.logical_cores;

    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    info.has_ht = (ebx >> 16) & 0xFF;

    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    info.has_avx2 = (ebx >> 5) & 1;
    info.has_avx512 = (ebx >> 16) & 1;
    info.has_amx = (ebx >> 22) & 1;

#if defined(__arm__) || defined(__aarch64__)
    info.has_neon = true;
#endif

    return info;
}

std::string simd_capabilities_string(const CPUInfo& info) {
    std::string caps;
    if (info.has_avx2) caps += "AVX2 ";
    if (info.has_avx512) caps += "AVX-512 ";
    if (info.has_amx) caps += "AMX ";
    if (info.has_neon) caps += "NEON ";
    if (caps.empty()) caps = "none";
    return caps;
}