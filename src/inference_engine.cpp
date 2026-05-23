#include "inference_engine.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>

InferenceEngine::InferenceEngine(const ModelConfig& config, Precision precision)
    : config_(config), precision_(precision), kv_stride_(config.dim) {
    WeightGenerator gen(42);

    uint64_t fp32_bytes = WeightGenerator::required_fp32_bytes(config);
    uint64_t int8_bytes = WeightGenerator::required_int8_bytes(config);

    (void)fp32_bytes;
    (void)int8_bytes;

    weights_fp32_ = gen.generate_weights_fp32(config);
    if (precision == Precision::INT8) {
        weights_int8_ = gen.generate_weights_int8(config);
    }

    uint64_t kv_total = 2ULL * config.dim * config.context_len * config.n_layers;
    kv_cache_.resize(kv_total, 0.0f);
}

float InferenceEngine::sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float InferenceEngine::tanh_custom(float x) {
    float e2 = expf(-2.0f * fabsf(x));
    return (1.0f - e2) / (1.0f + e2);
}

void InferenceEngine::run_prefill(int num_threads) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> input(config_.dim, 0.5f);
    std::vector<float> output(config_.dim, 0.0f);

    uint64_t attn_size = 4ULL * config_.dim * config_.dim;
    const float* w = weights_fp32_.data();

    for (uint32_t l = 0; l < config_.n_layers; ++l) {
        std::vector<float> q(config_.dim, 0.0f);
        std::vector<float> k(config_.dim, 0.0f);
        std::vector<float> v(config_.dim, 0.0f);

        const float* lw = w + l * 4ULL * config_.dim * config_.dim;

        for (uint32_t i = 0; i < config_.dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * lw[i * config_.dim + j];
            }
            q[i] = sum;
        }
        const float* lk = lw + config_.dim * config_.dim;
        for (uint32_t i = 0; i < config_.dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * lk[i * config_.dim + j];
            }
            k[i] = sum;
        }
        const float* lv = lw + 2ULL * config_.dim * config_.dim;
        for (uint32_t i = 0; i < config_.dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * lv[i * config_.dim + j];
            }
            v[i] = sum;
        }

        float score = 0.0f;
        for (uint32_t i = 0; i < config_.dim; ++i) {
            score += q[i] * k[i];
        }
        score /= sqrtf((float)config_.dim);
        score = tanh_custom(score);

        for (uint32_t i = 0; i < config_.dim; ++i) {
            output[i] += score * v[i];
        }

        uint64_t ffn_offset = 4ULL * config_.dim * config_.dim;
        const float* ffn_w = w + ffn_offset + l * 3ULL * config_.dim * config_.hidden_dim;

        std::vector<float> ffn_out(config_.dim, 0.0f);
        for (uint32_t i = 0; i < config_.dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * ffn_w[i * config_.dim + j];
            }
            float gate = sigmoid(sum);
            sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * ffn_w[config_.dim * config_.hidden_dim + i * config_.dim + j];
            }
            ffn_out[i] = gate * tanh_custom(sum);
        }
        for (uint32_t i = 0; i < config_.dim; ++i) {
            output[i] += ffn_out[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    last_stats_.ms_token = ms;
    last_stats_.tokens_per_sec = 128.0 / (ms / 1000.0);
}

void InferenceEngine::run_decode(int num_threads, uint32_t context_len) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> input(config_.dim, 0.3f);
    std::vector<float> output(config_.dim, 0.0f);

    uint64_t kv_layer_stride = 2ULL * config_.dim * context_len;
    uint64_t kv_total_needed = 2ULL * config_.dim * context_len * config_.n_layers;
    if (kv_cache_.size() < kv_total_needed) {
        kv_cache_.resize(kv_total_needed, 0.0f);
    }

    const float* w = weights_fp32_.data();

    for (uint32_t l = 0; l < config_.n_layers; ++l) {
        std::vector<float> q(config_.dim, 0.0f);
        float k_sum = 0.0f, v_sum = 0.0f;

        const float* lw = w + l * 4ULL * config_.dim * config_.dim;

        for (uint32_t i = 0; i < config_.dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * lw[i * config_.dim + j];
            }
            q[i] = sum;
        }

        for (uint32_t step = 0; step < context_len; ++step) {
            float* kv_layer_base = kv_cache_.data() + l * kv_layer_stride;
            float* k_base = kv_layer_base + step * config_.dim;
            float* v_base = kv_layer_base + config_.dim * context_len + step * config_.dim;

            const float* lk = lw + config_.dim * config_.dim;
            for (uint32_t i = 0; i < config_.dim; ++i) {
                float sum = 0.0f;
                for (uint32_t j = 0; j < config_.dim; ++j) {
                    sum += input[j] * lk[i * config_.dim + j];
                }
                k_base[i] = sum;
            }

            const float* lv = lw + 2ULL * config_.dim * config_.dim;
            for (uint32_t i = 0; i < config_.dim; ++i) {
                float sum = 0.0f;
                for (uint32_t j = 0; j < config_.dim; ++j) {
                    sum += input[j] * lv[i * config_.dim + j];
                }
                v_base[i] = sum;
            }

            for (uint32_t i = 0; i < config_.dim; ++i) {
                k_sum += q[i] * k_base[i];
                v_sum += k_base[i] * v_base[i];
            }
        }

        k_sum /= sqrtf((float)config_.dim);
        k_sum = tanh_custom(k_sum);

        for (uint32_t i = 0; i < config_.dim; ++i) {
            output[i] += k_sum * v_sum * 0.1f;
        }

        uint64_t ffn_offset = 4ULL * config_.dim * config_.dim;
        const float* ffn_w = w + ffn_offset + l * 3ULL * config_.dim * config_.hidden_dim;

        std::vector<float> ffn_out(config_.dim, 0.0f);
        for (uint32_t i = 0; i < config_.dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * ffn_w[i * config_.dim + j];
            }
            float gate = sigmoid(sum);
            sum = 0.0f;
            for (uint32_t j = 0; j < config_.dim; ++j) {
                sum += input[j] * ffn_w[config_.dim * config_.hidden_dim + i * config_.dim + j];
            }
            ffn_out[i] = gate * tanh_custom(sum);
        }
        for (uint32_t i = 0; i < config_.dim; ++i) {
            output[i] += ffn_out[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    last_stats_.ms_token = ms;
    last_stats_.tokens_per_sec = 1000.0 / ms;
}