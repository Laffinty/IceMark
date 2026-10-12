#pragma once
#include "model.hpp"
#include "weights.hpp"

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace icemark {

// Baseline dot products. Compiled without any /arch flag so they are valid on
// every x86-64 host.
float dot_f32_ref(const float* a, const float* b, size_t n);
int32_t dot_i8_ref(const int8_t* a, const int8_t* b, size_t n);

// AVX2+FMA variants. This translation unit is the only one compiled with
// /arch:AVX2 and must only ever be called after a runtime CPUID check.
bool cpu_supports_avx2_fma();
float dot_f32_avx2(const float* a, const float* b, size_t n);
int32_t dot_i8_avx2(const int8_t* a, const int8_t* b, size_t n);

enum class DotKind { Reference, Avx2 };

// Everything a single worker thread needs, allocated once in the constructor
// and reused for every measured iteration. No allocation, no zero-fill and no
// first-touch page fault is allowed to happen inside a timed region -- that
// was previously a per-thread 64MB fault storm plus ~100KB of vector churn
// per layer, both of which landed inside the measurement and both of which
// got worse with more threads.
struct Scratch {
    // activations, M x dim
    std::vector<float> x, norm, q, k, v, attn, proj;
    // FFN hidden, M x hidden_dim
    std::vector<float> gate, up, hidden;
    // attention scores: M*M for prefill, ctx for decode
    std::vector<float> scores;
    // KV cache: [K | V], each ctx x dim  (decode only, one buffer per thread).
    // Allocated lazily by prepare_decode() and only ever grown, never shrunk,
    // so a prefill-only run never pays for a multi-hundred-megabyte cache.
    std::vector<float> kv;

    // INT8 path: activations quantised with the same 1/128 scale as the
    // weights, so dot products accumulate in int32 exactly like a real
    // quantised kernel instead of doing a float multiply per element.
    std::vector<int8_t> xq, normq, hq;

    uint64_t bad_values = 0; // NaN / Inf counter, checked by the validity gate
};

// Multi-threaded inference kernel.
//
// Prefill runs M tokens through the full stack with causal self-attention, so
// raising M raises arithmetic intensity and walks the workload across the
// Roofline ridge point. Decode runs a single token per thread against a
// pre-filled private KV cache.
class Kernel {
public:
    Kernel(WeightStorePtr weights, uint32_t max_threads, uint32_t prefill_tokens);
    ~Kernel();

    // Wall-clock seconds for one full-stack pass on every worker thread.
    double run_prefill();
    // Must be called before run_decode(); fills and first-touches the caches.
    void prepare_decode(uint32_t context);
    double run_decode();

    uint32_t threads() const { return n_threads_; }
    uint32_t prefill_tokens() const { return prefill_tokens_; }
    uint32_t context() const { return context_; }

    // 2 FLOP per MAC. Weights are read exactly once per pass.
    double prefill_flops() const;
    // Weight bytes + per-thread KV bytes, all of them actually read.
    double decode_bytes() const;
    double prefill_bytes() const;

    uint64_t bad_values() const;

    // Cross-checks the dispatched kernels against the reference ones.
    // Returns the worst relative difference found (0.0 when within tolerance).
    double verify_dispatch(double tolerance);

    DotKind dot_kind() const { return dot_; }
    const char* dot_kind_name() const {
        return dot_ == DotKind::Avx2 ? "AVX2+FMA" : "scalar reference";
    }

    Precision precision() const { return precision_; }
    const ModelConfig& config() const { return cfg_; }

private:
    void thread_prefill(int tid);
    void thread_decode(int tid);
    void worker(uint32_t tid);
    double run_pass(bool prefill);

    struct DotFns {
        float (*f32)(const float*, const float*, size_t);
        int32_t (*i8)(const int8_t*, const int8_t*, size_t);
    };

    ModelConfig cfg_;
    Precision precision_;
    WeightStorePtr weights_;
    uint32_t n_threads_;
    uint32_t prefill_tokens_;
    uint32_t context_ = 0;
    uint32_t kv_capacity_ = 0;

    std::vector<Scratch> scratch_;
    DotFns dots_{nullptr, nullptr};
    DotKind dot_ = DotKind::Reference;

    // Persistent worker pool. See the comment above Kernel::run_pass for why
    // the main thread must sleep rather than spin.
    std::vector<std::thread> pool_;
    std::mutex m_;
    std::condition_variable cv_work_;
    uint64_t generation_ = 0;
    uint32_t ready_ = 0;
    uint32_t finished_ = 0;
    bool go_ = false;
    bool stop_ = false;
    bool current_pass_prefill_ = false;
};

} // namespace icemark