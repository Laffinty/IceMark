#include "kernel.hpp"
#include "weights.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#endif

namespace icemark {
namespace {

using DotF32 = float (*)(const float*, const float*, size_t);
using DotI8 = int32_t (*)(const int8_t*, const int8_t*, size_t);

struct DotFns {
    DotF32 f32;
    DotI8 i8;
};

constexpr float kRmsEps = 1e-5f;

inline double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

void rmsnorm(const float* x, float* out, uint32_t rows, uint32_t dim) {
    const float inv_dim = 1.0f / (float)dim;
    for (uint32_t r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * dim;
        float* orow = out + (size_t)r * dim;
        float ss = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) ss += xr[i] * xr[i];
        const float inv = 1.0f / sqrtf(ss * inv_dim + kRmsEps);
        for (uint32_t i = 0; i < dim; ++i) orow[i] = xr[i] * inv;
    }
}

void quantize_i8(const float* x, int8_t* out, uint32_t rows, uint32_t dim) {
    for (uint64_t i = 0, n = (uint64_t)rows * dim; i < n; ++i) {
        float v = x[i] * 128.0f;
        int32_t q = (int32_t)lrintf(v);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[i] = (int8_t)q;
    }
}

// Max-shift softmax; scores[] is left normalised in place.
void softmax(float* s, uint32_t n) {
    float mx = s[0];
    for (uint32_t i = 1; i < n; ++i) mx = std::max(mx, s[i]);
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        s[i] = expf(s[i] - mx);
        sum += s[i];
    }
    const float inv = 1.0f / sum;
    for (uint32_t i = 0; i < n; ++i) s[i] *= inv;
}

} // namespace

bool cpu_supports_avx2_fma() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int r[4] = {0};
    __cpuid(r, 0);
    if (r[0] < 7) return false;

    __cpuidex(r, 1, 0);
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    const bool fma = (r[2] & (1 << 12)) != 0;
    if (!osxsave || !fma) return false;

    // The OS must actually have enabled the AVX register state, otherwise
    // YMM use traps.
    const unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;

    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;  // AVX2
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------

Kernel::Kernel(WeightStorePtr weights, uint32_t max_threads, uint32_t prefill_tokens)
    : cfg_(weights->cfg), precision_(weights->precision), weights_(std::move(weights)),
      n_threads_(std::max(1u, max_threads)), prefill_tokens_(std::max(1u, prefill_tokens)) {
    const bool avx2 = cpu_supports_avx2_fma();
    DotFns fns;
    if (avx2) {
        fns.f32 = &dot_f32_avx2;
        fns.i8 = &dot_i8_avx2;
        dot_ = DotKind::Avx2;
    } else {
        fns.f32 = &dot_f32_ref;
        fns.i8 = &dot_i8_ref;
        dot_ = DotKind::Reference;
    }
    dots_ = fns;

    // Activation scratch is sized for the largest workload this kernel will
    // see. Allocating here -- not inside a timed region -- is what keeps page
    // faults and the allocator out of the measurement. The KV cache is NOT
    // allocated here: see prepare_decode().
    const uint32_t M = prefill_tokens_;
    const uint32_t d = cfg_.dim;
    const uint32_t h = cfg_.hidden_dim;
    const uint32_t score_cap = std::max<uint32_t>(M * M, 1u);

    scratch_.resize(n_threads_);
    for (auto& s : scratch_) {
        s.x.resize((size_t)M * d);
        s.norm.resize((size_t)M * d);
        s.q.resize((size_t)M * d);
        s.k.resize((size_t)M * d);
        s.v.resize((size_t)M * d);
        s.attn.resize((size_t)M * d);
        s.proj.resize((size_t)M * d);
        s.gate.resize((size_t)M * h);
        s.up.resize((size_t)M * h);
        s.hidden.resize((size_t)M * h);
        s.scores.resize(score_cap);
        s.xq.resize((size_t)M * d);
        s.normq.resize((size_t)M * d);
        s.hq.resize((size_t)M * h);
    }

    for (uint32_t t = 0; t < n_threads_; ++t) {
        pool_.emplace_back([this, t] { worker(t); });
    }
}

Kernel::~Kernel() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
        cv_work_.notify_all();
    }
    for (auto& th : pool_) {
        if (th.joinable()) th.join();
    }
}

uint64_t Kernel::bad_values() const {
    uint64_t n = 0;
    for (const auto& s : scratch_) n += s.bad_values;
    return n;
}

double Kernel::prefill_flops() const {
    return 2.0 * (double)cfg_.layer_params() * cfg_.n_layers *
           (double)prefill_tokens_ * (double)n_threads_;
}

double Kernel::prefill_bytes() const {
    return (double)cfg_.weight_bytes(precision_) * (double)n_threads_;
}

double Kernel::decode_bytes() const {
    const double kv = (double)2 * (double)context_ * (double)cfg_.dim * 4.0;
    return ((double)cfg_.weight_bytes(precision_) + kv * (double)cfg_.n_layers) *
           (double)n_threads_;
}

// ---------------------------------------------------------------------------
// Per-thread forward passes
// ---------------------------------------------------------------------------

void Kernel::thread_prefill(int tid) {
    Scratch& s = scratch_[tid];
    const uint32_t d = cfg_.dim;
    const uint32_t h = cfg_.hidden_dim;
    const uint32_t M = prefill_tokens_;
    const uint64_t layer_params = cfg_.layer_params();
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    const bool i8 = precision_ == Precision::INT8;
    const float* wbase = weights_->fp32.data();
    const int8_t* wbase8 = weights_->int8.data();

    std::fill(s.x.begin(), s.x.end(), 0.35f);

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const float* wf = wbase + (size_t)l * layer_params;
        const int8_t* wi = wbase8 + (size_t)l * layer_params;
        const float* Wq = wf;
        const float* Wk = wf + (size_t)d * d;
        const float* Wv = wf + 2 * (size_t)d * d;
        const float* Wo = wf + 3 * (size_t)d * d;
        const float* Wg = wf + 4 * (size_t)d * d;
        const float* Wu = Wg + (size_t)d * h;
        const float* Wdn = Wu + (size_t)d * h;

        rmsnorm(s.x.data(), s.norm.data(), M, d);

        // QKV projection, three weight rows fused so the activation row stays
        // resident in L1 across all three dots.
        if (i8) {
            quantize_i8(s.norm.data(), s.normq.data(), M, d);
            const int8_t* Wi_q = wi;
            const int8_t* Wi_k = wi + (size_t)d * d;
            const int8_t* Wi_v = wi + 2 * (size_t)d * d;
            for (uint32_t n = 0; n < d; ++n) {
                const int8_t* wq = Wi_q + (size_t)n * d;
                const int8_t* wk = Wi_k + (size_t)n * d;
                const int8_t* wv = Wi_v + (size_t)n * d;
                for (uint32_t m = 0; m < M; ++m) {
                    const int8_t* xq = s.normq.data() + (size_t)m * d;
                    s.q[(size_t)m * d + n] = dots_.i8(xq, wq, d) * kInt8Scale;
                    s.k[(size_t)m * d + n] = dots_.i8(xq, wk, d) * kInt8Scale;
                    s.v[(size_t)m * d + n] = dots_.i8(xq, wv, d) * kInt8Scale;
                }
            }
        } else {
            for (uint32_t n = 0; n < d; ++n) {
                const float* wq = Wq + (size_t)n * d;
                const float* wk = Wk + (size_t)n * d;
                const float* wv = Wv + (size_t)n * d;
                for (uint32_t m = 0; m < M; ++m) {
                    const float* xr = s.norm.data() + (size_t)m * d;
                    s.q[(size_t)m * d + n] = dots_.f32(xr, wq, d);
                    s.k[(size_t)m * d + n] = dots_.f32(xr, wk, d);
                    s.v[(size_t)m * d + n] = dots_.f32(xr, wv, d);
                }
            }
        }

        // Causal self-attention over the M prompt tokens.
        for (uint32_t m = 0; m < M; ++m) {
            float* sc = s.scores.data();
            const float* qm = s.q.data() + (size_t)m * d;
            for (uint32_t t = 0; t <= m; ++t) {
                sc[t] = dots_.f32(qm, s.k.data() + (size_t)t * d, d) * inv_sqrt_d;
            }
            softmax(sc, m + 1);
            float* ao = s.attn.data() + (size_t)m * d;
            std::memset(ao, 0, (size_t)d * sizeof(float));
            for (uint32_t t = 0; t <= m; ++t) {
                const float a = sc[t];
                const float* vr = s.v.data() + (size_t)t * d;
                for (uint32_t i = 0; i < d; ++i) ao[i] += a * vr[i];
            }
        }

        // Output projection + residual.
        if (i8) {
            quantize_i8(s.attn.data(), s.xq.data(), M, d);
            const int8_t* Wi_o = wi + 3 * (size_t)d * d;
            for (uint32_t n = 0; n < d; ++n) {
                const int8_t* wo = Wi_o + (size_t)n * d;
                for (uint32_t m = 0; m < M; ++m) {
                    s.proj[(size_t)m * d + n] =
                        dots_.i8(s.xq.data() + (size_t)m * d, wo, d) * kInt8Scale;
                }
            }
        } else {
            for (uint32_t n = 0; n < d; ++n) {
                const float* wo = Wo + (size_t)n * d;
                for (uint32_t m = 0; m < M; ++m) {
                    s.proj[(size_t)m * d + n] =
                        dots_.f32(s.attn.data() + (size_t)m * d, wo, d);
                }
            }
        }
        for (size_t i = 0, e = (size_t)M * d; i < e; ++i) s.x[i] += s.proj[i];

        // Feed-forward block.
        rmsnorm(s.x.data(), s.norm.data(), M, d);

        if (i8) {
            quantize_i8(s.norm.data(), s.normq.data(), M, d);
            const int8_t* Wi_g = wi + 4 * (size_t)d * d;
            const int8_t* Wi_u = Wi_g + (size_t)d * h;
            for (uint32_t j = 0; j < h; ++j) {
                const int8_t* wg = Wi_g + (size_t)j * d;
                const int8_t* wu = Wi_u + (size_t)j * d;
                for (uint32_t m = 0; m < M; ++m) {
                    const int8_t* xq = s.normq.data() + (size_t)m * d;
                    float sg = dots_.i8(xq, wg, d) * kInt8Scale;
                    float su = dots_.i8(xq, wu, d) * kInt8Scale;
                    s.gate[(size_t)m * h + j] = 1.0f / (1.0f + expf(-sg));
                    s.up[(size_t)m * h + j] = tanhf(su);
                }
            }
        } else {
            for (uint32_t j = 0; j < h; ++j) {
                const float* wg = Wg + (size_t)j * d;
                const float* wu = Wu + (size_t)j * d;
                for (uint32_t m = 0; m < M; ++m) {
                    const float* xr = s.norm.data() + (size_t)m * d;
                    float sg = dots_.f32(xr, wg, d);
                    float su = dots_.f32(xr, wu, d);
                    s.gate[(size_t)m * h + j] = 1.0f / (1.0f + expf(-sg));
                    s.up[(size_t)m * h + j] = tanhf(su);
                }
            }
        }

        for (size_t i = 0, e = (size_t)M * h; i < e; ++i) {
            s.hidden[i] = s.gate[i] * s.up[i];
        }

        // Wdown is [dim x hidden] and is consumed one row at a time, which is
        // the access order the original kernel got wrong: it walked this
        // matrix with a 32KB stride and that single mistake cost 11x.
        if (i8) {
            quantize_i8(s.hidden.data(), s.hq.data(), M, h);
            const int8_t* Wi_dn = wi + 4 * (size_t)d * d + 2 * (size_t)d * h;
            for (uint32_t i = 0; i < d; ++i) {
                const int8_t* wd = Wi_dn + (size_t)i * h;
                for (uint32_t m = 0; m < M; ++m) {
                    s.proj[(size_t)m * d + i] =
                        dots_.i8(s.hq.data() + (size_t)m * h, wd, h) * kInt8Scale;
                }
            }
        } else {
            for (uint32_t i = 0; i < d; ++i) {
                const float* wd = Wdn + (size_t)i * h;
                for (uint32_t m = 0; m < M; ++m) {
                    s.proj[(size_t)m * d + i] =
                        dots_.f32(s.hidden.data() + (size_t)m * h, wd, h);
                }
            }
        }
        for (size_t i = 0, e = (size_t)M * d; i < e; ++i) s.x[i] += s.proj[i];

        for (size_t i = 0, e = (size_t)M * d; i < e; ++i) {
            if (!std::isfinite(s.x[i])) ++s.bad_values;
        }
    }
}

void Kernel::thread_decode(int tid) {
    Scratch& s = scratch_[tid];
    const uint32_t d = cfg_.dim;
    const uint32_t h = cfg_.hidden_dim;
    const uint32_t ctx = context_;
    const uint64_t layer_params = cfg_.layer_params();
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    const bool i8 = precision_ == Precision::INT8;
    const float* wbase = weights_->fp32.data();
    const int8_t* wbase8 = weights_->int8.data();

    std::fill(s.x.begin(), s.x.end(), 0.35f);

    float* kbase = s.kv.data();
    float* vbase = kbase + (size_t)ctx * d;

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const float* wf = wbase + (size_t)l * layer_params;
        const int8_t* wi = wbase8 + (size_t)l * layer_params;
        const float* Wq = wf;
        const float* Wk = wf + (size_t)d * d;
        const float* Wv = wf + 2 * (size_t)d * d;
        const float* Wo = wf + 3 * (size_t)d * d;
        const float* Wg = wf + 4 * (size_t)d * d;
        const float* Wu = Wg + (size_t)d * h;
        const float* Wdn = Wu + (size_t)d * h;

        rmsnorm(s.x.data(), s.norm.data(), 1, d);

        if (i8) {
            quantize_i8(s.norm.data(), s.normq.data(), 1, d);
            const int8_t* xq = s.normq.data();
            for (uint32_t n = 0; n < d; ++n) {
                s.q[n] = dots_.i8(xq, wi + (size_t)n * d, d) * kInt8Scale;
                s.k[n] = dots_.i8(xq, wi + (size_t)d * d + (size_t)n * d, d) * kInt8Scale;
                s.v[n] = dots_.i8(xq, wi + 2 * (size_t)d * d + (size_t)n * d, d) * kInt8Scale;
            }
        } else {
            const float* xr = s.norm.data();
            for (uint32_t n = 0; n < d; ++n) {
                s.q[n] = dots_.f32(xr, Wq + (size_t)n * d, d);
                s.k[n] = dots_.f32(xr, Wk + (size_t)n * d, d);
                s.v[n] = dots_.f32(xr, Wv + (size_t)n * d, d);
            }
        }

        // Append this step to the private cache, then read all of it back.
        std::memcpy(kbase + (size_t)(ctx - 1) * d, s.k.data(), (size_t)d * sizeof(float));
        std::memcpy(vbase + (size_t)(ctx - 1) * d, s.v.data(), (size_t)d * sizeof(float));

        for (uint32_t t = 0; t < ctx; ++t) {
            s.scores[t] = dots_.f32(s.q.data(), kbase + (size_t)t * d, d) * inv_sqrt_d;
        }
        softmax(s.scores.data(), ctx);
        std::memset(s.attn.data(), 0, (size_t)d * sizeof(float));
        for (uint32_t t = 0; t < ctx; ++t) {
            const float a = s.scores[t];
            const float* vr = vbase + (size_t)t * d;
            float* ao = s.attn.data();
            for (uint32_t i = 0; i < d; ++i) ao[i] += a * vr[i];
        }

        if (i8) {
            quantize_i8(s.attn.data(), s.xq.data(), 1, d);
            const int8_t* Wo8 = wi + 3 * (size_t)d * d;
            for (uint32_t n = 0; n < d; ++n) {
                s.proj[n] = dots_.i8(s.xq.data(), Wo8 + (size_t)n * d, d) * kInt8Scale;
            }
        } else {
            for (uint32_t n = 0; n < d; ++n) {
                s.proj[n] = dots_.f32(s.attn.data(), Wo + (size_t)n * d, d);
            }
        }
        for (uint32_t i = 0; i < d; ++i) s.x[i] += s.proj[i];

        rmsnorm(s.x.data(), s.norm.data(), 1, d);

        if (i8) {
            quantize_i8(s.norm.data(), s.normq.data(), 1, d);
            const int8_t* xq = s.normq.data();
            const int8_t* Wi_g = wi + 4 * (size_t)d * d;
            const int8_t* Wi_u = Wi_g + (size_t)d * h;
            for (uint32_t j = 0; j < h; ++j) {
                s.gate[j] = 1.0f / (1.0f + expf(-dots_.i8(xq, Wi_g + (size_t)j * d, d) * kInt8Scale));
                s.up[j] = tanhf(dots_.i8(xq, Wi_u + (size_t)j * d, d) * kInt8Scale);
            }
        } else {
            const float* xr = s.norm.data();
            for (uint32_t j = 0; j < h; ++j) {
                s.gate[j] = 1.0f / (1.0f + expf(-dots_.f32(xr, Wg + (size_t)j * d, d)));
                s.up[j] = tanhf(dots_.f32(xr, Wu + (size_t)j * d, d));
            }
        }
        for (uint32_t j = 0; j < h; ++j) s.hidden[j] = s.gate[j] * s.up[j];

        if (i8) {
            quantize_i8(s.hidden.data(), s.hq.data(), 1, h);
            const int8_t* Wi_dn = wi + 4 * (size_t)d * d + 2 * (size_t)d * h;
            for (uint32_t i = 0; i < d; ++i) {
                s.proj[i] = dots_.i8(s.hq.data(), Wi_dn + (size_t)i * h, h) * kInt8Scale;
            }
        } else {
            for (uint32_t i = 0; i < d; ++i) {
                s.proj[i] = dots_.f32(s.hidden.data(), Wdn + (size_t)i * h, h);
            }
        }
        for (uint32_t i = 0; i < d; ++i) s.x[i] += s.proj[i];

        for (uint32_t i = 0; i < d; ++i) {
            if (!std::isfinite(s.x[i])) ++s.bad_values;
        }
    }
}

void Kernel::prepare_decode(uint32_t context) {
    context_ = context;
    const uint32_t d = cfg_.dim;

    // Grow-only, and only here. A prefill-only run therefore never allocates
    // a cache at all, and a decode sweep that walks a ladder of contexts pays
    // the largest allocation exactly once.
    if (context > kv_capacity_) {
        for (auto& s : scratch_) {
            s.kv.resize((size_t)2 * context * d);
            s.scores.resize(std::max(s.scores.size(), (size_t)context));
        }
        kv_capacity_ = context;
    }

    // Fill AND first-touch every cache here, outside any timed region. The
    // old harness allocated a zeroed KV cache inside the measured pass, so
    // each thread ate a fresh page-fault storm on every repeat and the cost
    // grew with thread count.
    for (auto& s : scratch_) {
        float* kb = s.kv.data();
        const size_t n = (size_t)2 * context_ * d;
        uint64_t st = 0x243F6A8885A308D3ull ^ (uint64_t)context_;
        for (size_t i = 0; i < n; ++i) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            kb[i] = (float)((st >> 33) / 16777216.0) * 2.0f - 1.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestration
//
// The workers live for the lifetime of the Kernel. Earlier versions spawned a
// fresh std::thread per pass and had the main thread spin on yield() until all
// workers reported ready -- on a 12-thread machine that is 13 runnable threads
// on 12 logical CPUs, and the amount of stealing varied from pass to pass.
// That showed up as +-20% noise at 4 and 12 threads while every single-thread
// measurement sat under 5%. The pool below keeps the main thread asleep on a
// condition variable and removes per-pass thread creation entirely.
// ---------------------------------------------------------------------------

double Kernel::run_pass(bool prefill) {
    double t0 = 0.0, t1 = 0.0;
    current_pass_prefill_ = prefill;

    {
        std::unique_lock<std::mutex> lk(m_);
        ++generation_;
        ready_ = 0;
        finished_ = 0;
        go_ = false;
        cv_work_.notify_all();
        cv_work_.wait(lk, [this] { return ready_ == n_threads_; });
        t0 = now_s();
        go_ = true;
        cv_work_.notify_all();
        cv_work_.wait(lk, [this] { return finished_ == n_threads_; });
    }
    t1 = now_s();
    return t1 - t0;
}

double Kernel::run_prefill() { return run_pass(true); }
double Kernel::run_decode() { return run_pass(false); }

void Kernel::worker(uint32_t tid) {
    uint64_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lk(m_);
        cv_work_.wait(lk, [this, &seen] { return stop_ || (generation_ != seen && !go_); });
        if (stop_) return;
        seen = generation_;
        ++ready_;
        cv_work_.notify_all();
        cv_work_.wait(lk, [this] { return stop_ || go_; });
        if (stop_) return;
        lk.unlock();

        const bool prefill = current_pass_prefill_;
        if (prefill) thread_prefill((int)tid);
        else thread_decode((int)tid);

        lk.lock();
        ++finished_;
        cv_work_.notify_all();
    }
}

double Kernel::verify_dispatch(double tolerance) {
    const size_t n = 4096;
    std::vector<float> a(n), b(n);
    std::vector<int8_t> a8(n), b8(n);
    uint64_t st = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; ++i) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        a[i] = ((float)((st >> 33) / 16777216.0) * 2.0f - 1.0f);
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        b[i] = ((float)((st >> 33) / 16777216.0) * 2.0f - 1.0f);
        a8[i] = (int8_t)(uint8_t)(st >> 56);
        b8[i] = (int8_t)(uint8_t)(st >> 56);
    }

    const float f_ref = dot_f32_ref(a.data(), b.data(), n);
    const float f_dis = (dot_ == DotKind::Avx2) ? dot_f32_avx2(a.data(), b.data(), n) : f_ref;
    const float f_err = fabsf(f_ref - f_dis) / std::max(1e-6f, fabsf(f_ref));

    const int32_t i_ref = dot_i8_ref(a8.data(), b8.data(), n);
    const int32_t i_dis = (dot_ == DotKind::Avx2) ? dot_i8_avx2(a8.data(), b8.data(), n) : i_ref;
    const float i_err = (float)std::abs(i_ref - i_dis) / (float)std::max(1, std::abs(i_ref));

    return std::max(f_err, i_err) > tolerance ? std::max(f_err, i_err) : 0.0;
}

} // namespace icemark