#include "weight_generator.hpp"
#include <algorithm>

WeightGenerator::WeightGenerator(uint64_t seed) : seed_(seed) {}

uint64_t WeightGenerator::lcg_next(uint64_t x) {
    return x * 6364136223846793005ULL + 1442695040888963407ULL;
}

float WeightGenerator::lcg_float(uint64_t& state) {
    state = lcg_next(state);
    return (float)(state >> 33) / 16777216.0f;
}

int8_t WeightGenerator::lcg_int8(uint64_t& state) {
    state = lcg_next(state);
    return (int8_t)(state >> 56);
}

uint64_t WeightGenerator::required_fp32_bytes(const ModelConfig& config) {
    uint64_t attn = 4ULL * config.dim * config.dim * config.n_layers;
    uint64_t ffn = 3ULL * config.dim * config.hidden_dim * config.n_layers;
    return (attn + ffn) * sizeof(float);
}

uint64_t WeightGenerator::required_int8_bytes(const ModelConfig& config) {
    uint64_t attn = 4ULL * config.dim * config.dim * config.n_layers;
    uint64_t ffn = 3ULL * config.dim * config.hidden_dim * config.n_layers;
    return attn + ffn;
}

std::vector<float> WeightGenerator::generate_weights_fp32(const ModelConfig& config) {
    uint64_t total = (4ULL * config.dim * config.dim + 3ULL * config.dim * config.hidden_dim) * config.n_layers;
    std::vector<float> w(total);
    uint64_t state = seed_;
    for (uint64_t i = 0; i < total; ++i) {
        w[i] = lcg_float(state) * 2.0f - 1.0f;
    }
    return w;
}

std::vector<int8_t> WeightGenerator::generate_weights_int8(const ModelConfig& config) {
    uint64_t total = (4ULL * config.dim * config.dim + 3ULL * config.dim * config.hidden_dim) * config.n_layers;
    std::vector<int8_t> w(total);
    uint64_t state = seed_;
    for (uint64_t i = 0; i < total; ++i) {
        w[i] = lcg_int8(state);
    }
    return w;
}