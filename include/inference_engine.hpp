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

    void run_prefill(int num_threads);
    void run_decode(int num_threads, uint32_t context_len);

    InferenceStats get_last_stats() const { return last_stats_; }

private:
    ModelConfig config_;
    Precision precision_;

    std::vector<float> weights_fp32_;
    std::vector<int8_t> weights_int8_;

    std::vector<float> kv_cache_;
    uint32_t kv_stride_;

    InferenceStats last_stats_;

    void run_layer_fp32(const float* input, float* output,
                        const float* weights, uint32_t dim, uint32_t n_layers);
    void run_layer_int8(const float* input, float* output,
                        const int8_t* weights, const float* scales,
                        uint32_t dim, uint32_t n_layers);

    void compute_qkv(const float* input, float* q, float* k, float* v,
                     uint32_t seq_len, const float* weights);
    void compute_attention(float* q, const float* kv_cache, uint32_t context_len, uint32_t dim);
    void compute_ffn(const float* input, float* output, const float* weights, uint32_t dim);
    float sigmoid(float x);
    float tanh_custom(float x);
};