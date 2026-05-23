#pragma once
#include <cstdint>
#include <vector>

enum class Precision { FP32, INT8 };

struct ModelConfig {
    uint32_t dim;        // model dimension
    uint32_t n_layers;   // number of layers
    uint32_t hidden_dim; // FFN hidden dimension
    uint32_t context_len;
};

class WeightGenerator {
public:
    WeightGenerator(uint64_t seed = 42);

    std::vector<float> generate_weights_fp32(const ModelConfig& config);
    std::vector<int8_t> generate_weights_int8(const ModelConfig& config);

    static uint64_t required_fp32_bytes(const ModelConfig& config);
    static uint64_t required_int8_bytes(const ModelConfig& config);

private:
    uint64_t seed_;
    uint64_t lcg_next(uint64_t x);
    float lcg_float(uint64_t& state);
    int8_t lcg_int8(uint64_t& state);
};