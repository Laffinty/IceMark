// Baseline kernels. Deliberately compiled with no /arch flag so this
// translation unit stays valid on any x86-64 host.
#include "kernel.hpp"

#include <cmath>

namespace icemark {

float dot_f32_ref(const float* a, const float* b, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

int32_t dot_i8_ref(const int8_t* a, const int8_t* b, size_t n) {
    int32_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

} // namespace icemark