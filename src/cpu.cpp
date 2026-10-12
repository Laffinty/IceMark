#include "cpu.hpp"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#include <unistd.h>
#endif

namespace icemark {
namespace {

#if defined(_WIN32) || defined(_WIN64)
void cpuid_count(uint32_t leaf, uint32_t sub, int out[4]) {
    __cpuidex(out, (int)leaf, (int)sub);
}
uint64_t xgetbv0() { return (uint64_t)_xgetbv(0); }
#else
void cpuid_count(uint32_t leaf, uint32_t sub, int out[4]) {
    __cpuid_count(leaf, sub, out[0], out[1], out[2], out[3]);
}
uint64_t xgetbv0() {
    uint32_t eax, edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}
#endif

std::string trim(const char* p) {
    std::string s(p);
    size_t a = s.find_first_not_of(' ');
    size_t b = s.find_last_not_of(' ');
    if (a == std::string::npos) return {};
    return s.substr(a, b - a + 1);
}

} // namespace

CPUInfo detect_cpu() {
    CPUInfo info;
    info.logical_cores = std::thread::hardware_concurrency();
    if (info.logical_cores == 0) info.logical_cores = 4;

#if defined(_WIN32) || defined(_WIN64)
    // Ask Windows for the real core topology. The old CPUID leaf-0x0B path
    // returned the logical-processor count of the SMT level, which on a 6C/12T
    // part reported "12 physical cores" for a chip that has 6.
    {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len) {
            std::vector<unsigned char> buf(len);
            auto* p = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, p, &len)) {
                DWORD cores = 0;
                auto* it = reinterpret_cast<unsigned char*>(p);
                auto* end = it + len;
                while (it < end) {
                    auto* rec = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(it);
                    ++cores;
                    it += rec->Size;
                }
                if (cores > 0) {
                    info.physical_cores = cores;
                    info.has_ht = cores < info.logical_cores;
                }
            }
        }
    }
    if (info.physical_cores == 0) {
        info.physical_cores = info.logical_cores;
    }

    {
        HKEY k;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                          "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                          KEY_READ, &k) == ERROR_SUCCESS) {
            DWORD v = 0, sz = sizeof(v);
            if (RegQueryValueExA(k, "~MHz", nullptr, nullptr, (LPBYTE)&v, &sz) == ERROR_SUCCESS) {
                info.max_freq_mhz = v;
            }
            RegCloseKey(k);
        }
    }
#else
    info.physical_cores = info.logical_cores;
#endif

    int r[4] = {0, 0, 0, 0};
    cpuid_count(0, 0, r);
    const uint32_t max_leaf = (uint32_t)r[0];

    if (max_leaf >= 1) {
        cpuid_count(1, 0, r);
        info.has_ht = (r[3] >> 28) & 1;
        info.has_fma = (r[2] >> 12) & 1;

        const bool osxsave = (r[2] >> 27) & 1;
        const uint64_t xcr0 = osxsave ? xgetbv0() : 0;
        // bits 1|2 = SSE + AVX state, bits 5|6|7 = opmask/ZMM_hi256/Hi16_ZMM
        const bool os_avx = osxsave && (xcr0 & 0x6) == 0x6;
        const bool os_zmm = os_avx && (xcr0 & 0xE0) == 0xE0;

        if (max_leaf >= 7) {
            int r7[4] = {0, 0, 0, 0};
            cpuid_count(7, 0, r7);
            info.has_avx2 = os_avx && ((r7[1] >> 5) & 1);
            const bool avx512f = (r7[1] >> 16) & 1;
            info.has_avx512 = os_zmm && avx512f;
            // AMX-TILE is leaf 7 EBX bit 24 and AMX-INT8 is bit 25. The old
            // code read bit 22, which is neither.
            info.has_amx = os_zmm && (((r7[1] >> 24) & 1) && ((r7[1] >> 25) & 1));
        }

        // Brand string (leaves 0x80000002..4).
        char brand[49] = {0};
        uint32_t ext = 0x80000000u;
        cpuid_count(ext, 0, r);
        if ((uint32_t)r[0] >= 0x80000004u) {
            int* dst = reinterpret_cast<int*>(brand);
            for (uint32_t i = 0; i < 3; ++i) {
                cpuid_count(ext + 2 + i, 0, r);
                dst[i * 4 + 0] = r[0];
                dst[i * 4 + 1] = r[1];
                dst[i * 4 + 2] = r[2];
                dst[i * 4 + 3] = r[3];
            }
            info.brand = trim(brand);
        }
        if (info.max_freq_mhz == 0 && max_leaf >= 0x16) {
            int r16[4] = {0, 0, 0, 0};
            cpuid_count(0x16, 0, r16);
            if (r16[0] > 0) info.max_freq_mhz = r16[0];
        }
    }

#if defined(__arm__) || defined(__aarch64__)
    info.has_neon = true;
    info.brand = "ARM";
#endif

    return info;
}

std::string simd_capabilities_string(const CPUInfo& info) {
    std::string caps;
    auto add = [&caps](const char* s) {
        if (!caps.empty()) caps += " ";
        caps += s;
    };
    if (info.has_fma) add("FMA");
    if (info.has_avx2) add("AVX2");
    if (info.has_avx512) add("AVX-512");
    if (info.has_amx) add("AMX");
    if (info.has_neon) add("NEON");
    if (caps.empty()) caps = "scalar only";
    return caps;
}

MemoryInfo detect_memory() {
    MemoryInfo m;
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        m.total_bytes = st.ullTotalPhys;
        m.avail_bytes = st.ullAvailPhys;
    }
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long psz = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psz > 0) {
        m.total_bytes = (uint64_t)pages * (uint64_t)psz;
        m.avail_bytes = m.total_bytes / 2; // no portable figure available
    }
#endif
    return m;
}

} // namespace icemark