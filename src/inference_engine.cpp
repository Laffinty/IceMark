#include "inference_engine.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

InferenceEngine::InferenceEngine(const ModelConfig& config, Precision precision)
    : config_(config), precision_(precision) {
    WeightGenerator gen(42);
    weights_fp32_ = gen.generate_weights_fp32(config);
    if (precision == Precision::INT8) {
        weights_int8_ = gen.generate_weights_int8(config);
    }
}

float InferenceEngine::sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float InferenceEngine::tanh_approx(float x) {
    float e2 = expf(-2.0f * fabsf(x));
    return (1.0f - e2) / (1.0f + e2);
}

void InferenceEngine::softmax(float* data, uint32_t len) {
    float max_val = data[0];
    for (uint32_t i = 1; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < len; ++i) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for (uint32_t i = 0; i < len; ++i) {
        data[i] /= sum;
    }
}

void InferenceEngine::run_layer_fp32(const float* input, float* output, uint32_t layer_idx,
                                     float* kv_cache, uint32_t context_len, uint32_t current_step) {
    uint32_t d = config_.dim;
    uint32_t h = config_.hidden_dim;
    uint64_t layer_size = 4ULL * d * d + 3ULL * d * h;
    const float* w = weights_fp32_.data() + layer_idx * layer_size;
    const float* w_q = w;
    const float* w_k = w + d * d;
    const float* w_v = w + 2ULL * d * d;
    const float* w_gate = w + 4ULL * d * d;
    const float* w_up   = w + 4ULL * d * d + d * h;
    const float* w_down = w + 4ULL * d * d + 2ULL * d * h;

    // QKV projection: input(1 x dim) @ W(dim x dim) -> output(1 x dim)
    std::vector<float> q(d), k(d), v(d);
    for (uint32_t i = 0; i < d; ++i) {
        float sq = 0.0f, sk = 0.0f, sv = 0.0f;
        for (uint32_t j = 0; j < d; ++j) {
            sq += input[j] * w_q[i * d + j];
            sk += input[j] * w_k[i * d + j];
            sv += input[j] * w_v[i * d + j];
        }
        q[i] = sq; k[i] = sk; v[i] = sv;
    }

    // Write K, V to KV-Cache at current step
    if (kv_cache) {
        float* k_cache = kv_cache + current_step * d;
        float* v_cache = kv_cache + context_len * d + current_step * d;
        for (uint32_t i = 0; i < d; ++i) {
            k_cache[i] = k[i];
            v_cache[i] = v[i];
        }
    }

    // Attention: read all historical K/V from cache
    std::vector<float> attn_out(d, 0.0f);
    if (kv_cache && context_len > 0) {
        std::vector<float> scores(context_len);
        float scale = 1.0f / sqrtf((float)d);
        for (uint32_t step = 0; step < context_len; ++step) {
            float* k_step = kv_cache + step * d;
            float dot = 0.0f;
            for (uint32_t i = 0; i < d; ++i) {
                dot += q[i] * k_step[i];
            }
            scores[step] = dot * scale;
        }
        softmax(scores.data(), context_len);
        for (uint32_t step = 0; step < context_len; ++step) {
            float* v_step = kv_cache + context_len * d + step * d;
            float s = scores[step];
            for (uint32_t i = 0; i < d; ++i) {
                attn_out[i] += s * v_step[i];
            }
        }
    } else {
        for (uint32_t i = 0; i < d; ++i) {
            attn_out[i] = v[i];
        }
    }

    // FFN: gate/up (dim -> hidden_dim), down (hidden_dim -> dim)
    std::vector<float> gate(h), up_vec(h), ffn_out(d, 0.0f);
    for (uint32_t j = 0; j < h; ++j) {
        float sg = 0.0f, su = 0.0f;
        for (uint32_t i = 0; i < d; ++i) {
            sg += input[i] * w_gate[j * d + i];
            su += input[i] * w_up[j * d + i];
        }
        gate[j] = sigmoid(sg);
        up_vec[j] = tanh_approx(su);
    }
    for (uint32_t j = 0; j < h; ++j) {
        float val = gate[j] * up_vec[j];
        for (uint32_t i = 0; i < d; ++i) {
            ffn_out[i] += val * w_down[i * h + j];
        }
    }

    // Residual add
    for (uint32_t i = 0; i < d; ++i) {
        output[i] = attn_out[i] + ffn_out[i];
    }
}

void InferenceEngine::run_layer_int8(const float* input, float* output, uint32_t layer_idx,
                                     float* kv_cache, uint32_t context_len, uint32_t current_step) {
    uint32_t d = config_.dim;
    uint32_t h = config_.hidden_dim;
    uint64_t layer_size = 4ULL * d * d + 3ULL * d * h;
    const int8_t* w = weights_int8_.data() + layer_idx * layer_size;
    const int8_t* w_q = w;
    const int8_t* w_k = w + d * d;
    const int8_t* w_v = w + 2ULL * d * d;
    const int8_t* w_gate = w + 4ULL * d * d;
    const int8_t* w_up   = w + 4ULL * d * d + d * h;
    const int8_t* w_down = w + 4ULL * d * d + 2ULL * d * h;
    const float scale = 1.0f / 128.0f;

    std::vector<float> q(d), k(d), v(d);
    for (uint32_t i = 0; i < d; ++i) {
        float sq = 0.0f, sk = 0.0f, sv = 0.0f;
        for (uint32_t j = 0; j < d; ++j) {
            sq += input[j] * (float)w_q[i * d + j] * scale;
            sk += input[j] * (float)w_k[i * d + j] * scale;
            sv += input[j] * (float)w_v[i * d + j] * scale;
        }
        q[i] = sq; k[i] = sk; v[i] = sv;
    }

    if (kv_cache) {
        float* k_cache = kv_cache + current_step * d;
        float* v_cache = kv_cache + context_len * d + current_step * d;
        for (uint32_t i = 0; i < d; ++i) {
            k_cache[i] = k[i];
            v_cache[i] = v[i];
        }
    }

    std::vector<float> attn_out(d, 0.0f);
    if (kv_cache && context_len > 0) {
        std::vector<float> scores(context_len);
        float s = 1.0f / sqrtf((float)d);
        for (uint32_t step = 0; step < context_len; ++step) {
            float* k_step = kv_cache + step * d;
            float dot = 0.0f;
            for (uint32_t i = 0; i < d; ++i) {
                dot += q[i] * k_step[i];
            }
            scores[step] = dot * s;
        }
        softmax(scores.data(), context_len);
        for (uint32_t step = 0; step < context_len; ++step) {
            float* v_step = kv_cache + context_len * d + step * d;
            float sc = scores[step];
            for (uint32_t i = 0; i < d; ++i) {
                attn_out[i] += sc * v_step[i];
            }
        }
    } else {
        for (uint32_t i = 0; i < d; ++i) {
            attn_out[i] = v[i];
        }
    }

    std::vector<float> gate(h), up_vec(h), ffn_out(d, 0.0f);
    for (uint32_t j = 0; j < h; ++j) {
        float sg = 0.0f, su = 0.0f;
        for (uint32_t i = 0; i < d; ++i) {
            sg += input[i] * (float)w_gate[j * d + i] * scale;
            su += input[i] * (float)w_up[j * d + i] * scale;
        }
        gate[j] = sigmoid(sg);
        up_vec[j] = tanh_approx(su);
    }
    for (uint32_t j = 0; j < h; ++j) {
        float val = gate[j] * up_vec[j];
        for (uint32_t i = 0; i < d; ++i) {
            ffn_out[i] += val * (float)w_down[i * h + j] * scale;
        }
    }

    for (uint32_t i = 0; i < d; ++i) {
        output[i] = attn_out[i] + ffn_out[i];
    }
}

InferenceStats InferenceEngine::run_prefill(int num_threads) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this]() {
            std::vector<float> input(config_.dim, 0.5f);
            std::vector<float> output(config_.dim, 0.0f);
            for (int token = 0; token < 1; ++token) {
                for (uint32_t l = 0; l < config_.n_layers; ++l) {
                    if (precision_ == Precision::INT8) {
                        this->run_layer_int8(input.data(), output.data(), l, nullptr, 0, 0);
                    } else {
                        this->run_layer_fp32(input.data(), output.data(), l, nullptr, 0, 0);
                    }
                    std::swap(input, output);
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double total_tokens = 128.0 * num_threads;
    return {ms, total_tokens / (ms / 1000.0)};
}

InferenceStats InferenceEngine::run_decode(int num_threads, uint32_t context_len) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, context_len]() {
            std::vector<float> input(config_.dim, 0.3f);
            std::vector<float> output(config_.dim, 0.0f);
            std::vector<float> kv_cache(2ULL * context_len * config_.dim, 0.0f);

            for (uint32_t l = 0; l < config_.n_layers; ++l) {
                if (precision_ == Precision::INT8) {
                    this->run_layer_int8(input.data(), output.data(), l,
                                         kv_cache.data(), context_len, context_len - 1);
                } else {
                    this->run_layer_fp32(input.data(), output.data(), l,
                                         kv_cache.data(), context_len, context_len - 1);
                }
                std::swap(input, output);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return {ms / num_threads, 1000.0 * num_threads / ms};
}
