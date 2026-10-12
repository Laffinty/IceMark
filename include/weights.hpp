#pragma once
#include "model.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace icemark {

// Deterministic weight synthesis that does not depend on the thread count.
//
// Each layer draws from its own splitmix64-seeded LCG stream. Layers are
// therefore independent and can be generated concurrently while staying
// bit-identical to a serial run -- which the previous single sequential
// stream could not do, and which cost ~2s of serial startup for the large
// profile.
class WeightGenerator {
public:
    static constexpr uint64_t kSeed = 42;

    // FP32 weights map to [-1, 1].
    void generate_fp32(const ModelConfig& cfg, std::vector<float>& out) const;

    // INT8 weights span the full signed range. Runtime dequantisation uses
    // w_fp32 = w_int8 * (1/128), i.e. Jacob et al. 2018 symmetric per-tensor
    // quantisation. Activations are quantised with the same scale so the
    // dot products accumulate in int32 exactly as a real INT8 kernel would.
    void generate_int8(const ModelConfig& cfg, std::vector<int8_t>& out) const;

private:
    static uint64_t splitmix64(uint64_t x);
    static uint64_t lcg_next(uint64_t x) {
        return x * 6364136223846793005ull + 1442695040888963407ull;
    }
    // [0,1) from the top 31 bits, scaled to 2^24 -- reproducible on any
    // IEEE-754 host.
    static float lcg_unit(uint64_t& s) {
        s = lcg_next(s);
        return (float)(s >> 33) / 16777216.0f;
    }
    static int8_t lcg_i8(uint64_t& s) {
        s = lcg_next(s);
        return (int8_t)(uint8_t)(s >> 56);
    }
};

constexpr float kInt8Scale = 1.0f / 128.0f;

// Weights are identical for every thread count, so they are generated once and
// shared. Previously each Kernel regenerated its own multi-gigabyte copy, which
// both wasted startup time and multiplied peak memory by the number of live
// kernels -- enough on a 16GB machine to page.
struct WeightStore {
    ModelConfig cfg;
    Precision precision = Precision::FP32;
    std::vector<float> fp32;
    std::vector<int8_t> int8;
};

using WeightStorePtr = std::shared_ptr<const WeightStore>;

WeightStorePtr make_weights(const ModelConfig& cfg, Precision p);

} // namespace icemark