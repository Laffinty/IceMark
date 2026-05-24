#pragma once
#include "weight_generator.hpp"
#include <vector>
#include <cstdint>

struct InferenceStats {
    double ms_token;
    double tokens_per_sec;
};

class InferenceEngine {
public:
    InferenceEngine(const ModelConfig& config, Precision precision);

    InferenceStats run_prefill(int num_threads);
    InferenceStats run_decode(int num_threads, uint32_t context_len);

private:
    ModelConfig config_;
    Precision precision_;

    std::vector<float> weights_fp32_;
    std::vector<int8_t> weights_int8_;

    void run_layer_fp32(const float* input, float* output, uint32_t layer_idx,
                        float* kv_cache, uint32_t context_len, uint32_t current_step);
    void run_layer_int8(const float* input, float* output, uint32_t layer_idx,
                        float* kv_cache, uint32_t context_len, uint32_t current_step);

    static float sigmoid(float x);
    static float tanh_approx(float x);
    static void softmax(float* data, uint32_t len);
};
