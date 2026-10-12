#include "weights.hpp"

#include <algorithm>
#include <thread>

namespace icemark {

const char* profile_name(Profile p) {
    switch (p) {
        case Profile::Quick:    return "quick";
        case Profile::Standard: return "standard";
        case Profile::Deep:     return "deep";
    }
    return "standard";
}

// Sized so that one forward pass streams noticeably more than any current
// CPU's last-level cache, while a full GUI run still finishes in tens of
// seconds. dim=2048/L=16 is carried over from the original "small" preset,
// but that preset cost 4.29 GB of weights and ~90s; the default is now
// sized for interactive use.
ModelConfig profile_config(Profile p) {
    switch (p) {
        case Profile::Quick:    return {768,  6,  3072};
        case Profile::Standard: return {1280, 10, 5120};
        case Profile::Deep:     return {2048, 16, 8192};
    }
    return {1280, 10, 5120};
}

uint64_t WeightGenerator::splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

void WeightGenerator::generate_fp32(const ModelConfig& cfg, std::vector<float>& out) const {
    const uint64_t per_layer = cfg.layer_params();
    out.assign(per_layer * cfg.n_layers, 0.0f);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t layers = cfg.n_layers;
    const uint32_t nthreads = std::min<uint32_t>(hw, layers);

    auto work = [&](uint32_t lo, uint32_t hi) {
        for (uint32_t l = lo; l < hi; ++l) {
            uint64_t s = splitmix64(kSeed ^ (0x1234567ull * (l + 1)));
            float* dst = out.data() + (size_t)l * per_layer;
            for (uint64_t i = 0; i < per_layer; ++i) {
                dst[i] = lcg_unit(s) * 2.0f - 1.0f;
            }
        }
    };

    std::vector<std::thread> pool;
    for (uint32_t t = 1; t < nthreads; ++t) {
        pool.emplace_back(work, t * layers / nthreads, (t + 1) * layers / nthreads);
    }
    work(0, layers / nthreads);
    for (auto& th : pool) th.join();
}

void WeightGenerator::generate_int8(const ModelConfig& cfg, std::vector<int8_t>& out) const {
    const uint64_t per_layer = cfg.layer_params();
    out.assign(per_layer * cfg.n_layers, 0);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t layers = cfg.n_layers;
    const uint32_t nthreads = std::min<uint32_t>(hw, layers);

    auto work = [&](uint32_t lo, uint32_t hi) {
        for (uint32_t l = lo; l < hi; ++l) {
            uint64_t s = splitmix64(kSeed ^ (0x1234567ull * (l + 1)));
            int8_t* dst = out.data() + (size_t)l * per_layer;
            for (uint64_t i = 0; i < per_layer; ++i) {
                dst[i] = lcg_i8(s);
            }
        }
    };

    std::vector<std::thread> pool;
    for (uint32_t t = 1; t < nthreads; ++t) {
        pool.emplace_back(work, t * layers / nthreads, (t + 1) * layers / nthreads);
    }
    work(0, layers / nthreads);
    for (auto& th : pool) th.join();
}

WeightStorePtr make_weights(const ModelConfig& cfg, Precision p) {
    auto store = std::make_shared<WeightStore>();
    store->cfg = cfg;
    store->precision = p;
    WeightGenerator gen;
    if (p == Precision::FP32) {
        gen.generate_fp32(cfg, store->fp32);
    } else {
        gen.generate_int8(cfg, store->int8);
    }
    return store;
}

} // namespace icemark