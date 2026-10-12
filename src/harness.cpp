#include "harness.hpp"
#include "weights.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>

namespace icemark {
namespace {

double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Boost clocks and settle the governor before anything is timed. The very
// first loop a fresh process runs executes at whatever clock the governor
// happened to be parked at, which showed up as a bandwidth reference of
// 4.1 GB/s against a steady-state 14.5 GB/s on the same machine.
void spin_up_cpu(double seconds) {
    const double t0 = now_s();
    double a = 1.0000001, b = 1.0;
    while (now_s() - t0 < seconds) {
        for (int i = 0; i < 50000; ++i) {
            a = a * b + 1e-9;
            b = b + a * 1e-12;
        }
    }
    (void)a;
    (void)b;
}

// Samples until the dispersion is good enough, the repeat cap is reached, or
// the time budget is spent. A fixed repeat count cannot distinguish "the
// machine is noisy" from "the machine is slow": both just return three
// numbers, and the median of three noisy samples still reports large error
// bars that a reader has no way to interpret.
template <typename F>
Stat sample_adaptive(F&& once, uint32_t min_n, uint32_t max_n, double target_noise,
                     double max_seconds) {
    std::vector<double> v;
    v.reserve(max_n);
    const double t0 = now_s();
    Stat st;
    for (uint32_t i = 0; i < max_n; ++i) {
        v.push_back(once());
        if (i + 1 >= min_n) {
            st = summarize(v);
            if (st.rel_noise <= target_noise) break;
            if (now_s() - t0 >= max_seconds) break;
        }
    }
    if (v.empty()) return st;
    return summarize(v);
}

constexpr double kTargetNoise = 0.05;
constexpr double kSampleSeconds = 0.15;

// A single pass at the quick profile is ~20ms, which is short enough that one
// context switch or one boost dip moves the median more than the effect being
// measured. Each sample therefore runs several back-to-back passes so the
// timer jitter is spread across the whole sample instead of dominating it.
uint32_t inner_reps_for(double pass_seconds, double target_seconds) {
    if (!(pass_seconds > 0.0)) return 1;
    const double n = target_seconds / pass_seconds;
    if (n <= 1.0) return 1;
    if (n >= 64.0) return 64;
    return (uint32_t)(n + 0.5);
}

constexpr uint32_t kMLadderMax = 32;

std::vector<uint32_t> m_ladder() {
    std::vector<uint32_t> v;
    for (uint32_t m = 1; m <= kMLadderMax; m *= 2) v.push_back(m);
    return v;
}

std::vector<uint32_t> thread_ladder(uint32_t max_threads) {
    std::vector<uint32_t> v;
    for (uint32_t t = 1; t <= max_threads; t *= 2) v.push_back(t);
    if (v.empty()) v.push_back(1);
    if (v.back() != max_threads) v.push_back(max_threads);
    return v;
}

// Geometric ladder from a small context up to `hi`, never repeating a value.
std::vector<uint32_t> ctx_ladder(uint32_t hi, size_t count) {
    std::vector<uint32_t> v;
    if (hi < 64) return {std::max<uint32_t>(hi, 16)};
    const double lo = std::max(256.0, (double)hi / 64.0);
    for (size_t i = 0; i < count; ++i) {
        double f = std::pow((double)hi / lo, (double)i / (double)(count - 1));
        uint32_t c = (uint32_t)(lo * f);
        c = std::max<uint32_t>(64, (c / 64) * 64);
        if (v.empty() || c > v.back() + 63) v.push_back(c);
    }
    v.back() = std::max(v.back(), std::min<uint32_t>(hi, v.back()));
    return v;
}

} // namespace

Stat summarize(std::vector<double> samples) {
    Stat st;
    if (samples.empty()) return st;
    std::sort(samples.begin(), samples.end());

    st.n = (int)samples.size();
    st.median = samples[samples.size() / 2];

    std::vector<double> dev;
    dev.reserve(samples.size());
    for (double s : samples) dev.push_back(std::fabs(s - st.median));
    std::sort(dev.begin(), dev.end());
    st.mad = dev[dev.size() / 2];

    // Reject gross outliers on both sides. A floor of 5% of the median keeps a
// tightly-clustered set (MAD == 0) from rejecting everything above its own
// median, which is what a naive MAD threshold does with five identical
// samples.
const double sigma = std::max(1.4826 * st.mad, 0.05 * std::fabs(st.median));
const double thresh = 3.0 * sigma;
std::vector<double> kept;
for (double s : samples) {
    if (std::fabs(s - st.median) > thresh) ++st.rejected;
    else kept.push_back(s);
}
    if (!kept.empty()) {
        st.median = kept[kept.size() / 2];
        st.n = (int)kept.size();
    }
    std::sort(kept.begin(), kept.end());
    st.lo = kept.empty() ? 0.0 : kept.front();
    st.hi = kept.empty() ? 0.0 : kept.back();
    st.rel_noise = st.median > 0.0 ? (1.4826 * st.mad) / st.median : 0.0;
    return st;
}

double measure_stream_bandwidth_gb_s(size_t buffer_bytes) {
    std::vector<unsigned char> buf(buffer_bytes);
    // First-touch outside the timed region.
    std::memset(buf.data(), 1, buffer_bytes);

    double best = 0.0;
    uint64_t sink = 0;
    for (int pass = 0; pass < 3; ++pass) {
        const double t0 = now_s();
        uint64_t acc = 0;
        const unsigned char* p = buf.data();
        for (size_t i = 0; i < buffer_bytes; i += 64) {
            uint64_t v;
            std::memcpy(&v, p + i, sizeof(v));
            acc += v;
        }
        const double t1 = now_s();
        sink += acc;
        if (t1 > t0) best = std::max(best, (double)buffer_bytes / (t1 - t0) / 1e9);
    }
    if (sink == 0xDEADBEEFull) best += 1.0;  // never taken; defeats elision
    return best;
}

Result Harness::run() {
    try {
        return run_impl();
    } catch (const std::bad_alloc&) {
        Result r;
        r.opt = opt_;
        r.error = "内存不足：测量中途无法分配工作集。请关闭其它占用内存的程序，"
                  "或改用更小的模型规模 / 更少的线程。";
        r.total_seconds = 0.0;
        return r;
    } catch (const std::exception& e) {
        Result r;
        r.opt = opt_;
        r.error = std::string("测量异常终止：") + e.what();
        return r;
    }
}

Result Harness::run_impl() {
    const double t_start = now_s();
    Result r;
    r.opt = opt_;
    r.cpu = detect_cpu();
    r.mem = detect_memory();
    r.model = profile_config(opt_.profile);
    r.dot_kind = cpu_supports_avx2_fma() ? "AVX2+FMA" : "scalar reference";

    report("自检：提升时钟并测量顺序访存带宽", 0.02);
    spin_up_cpu(0.30);
    const uint32_t d = r.model.dim;

    // Yardstick buffer: several times any plausible L3, capped so a huge
    // machine does not spend the whole budget on the self-check.
    size_t yard = 512ull << 20;
    yard = std::min<size_t>(yard, (size_t)std::max<uint64_t>(64ull << 20, r.mem.avail_bytes / 8));
    r.stream_gb_s = measure_stream_bandwidth_gb_s(yard);
    r.check.stream_gb_s = r.stream_gb_s;

    // ---- thread budget -----------------------------------------------------
    // The old heuristic charged every thread for n_layers KV caches and for a
    // context of 4096, while the benchmark actually ran contexts of 128/512
    // and one cache per thread. On a 15.9GB / 12-thread machine it silently
    // capped the run at 8 threads and reported the resulting curve as if it
    // were the machine's limit.
    uint32_t max_threads = opt_.max_threads ? opt_.max_threads : r.cpu.logical_cores;
    max_threads = std::max(1u, std::min(max_threads, r.cpu.logical_cores));

    // Budget for the decode sweep must leave room for the weight sets that stay
    // resident for the whole run, not just the caches.
    const uint64_t resident = r.model.weight_bytes(Precision::FP32) *
                              (opt_.with_int8 ? 2 : 1);
    const uint64_t budget = r.mem.avail_bytes > resident + (256ull << 20)
                                ? (uint64_t)((double)(r.mem.avail_bytes - resident) * 0.60)
                                : (256ull << 20);
    const uint64_t bytes_per_ctx = 2ull * d * 4ull;  // K and V rows
    uint32_t ctx_max_mem = (uint32_t)std::min<uint64_t>(
        65536ull, budget / ((uint64_t)max_threads * bytes_per_ctx));
    ctx_max_mem = std::max<uint32_t>(ctx_max_mem, 256);

    // Context where KV traffic equals weight traffic -- the point at which the
    // decode result is actually measuring the cache instead of the weights.
    const uint32_t ctx_parity = (uint32_t)std::min<uint64_t>(
        65536ull, (r.model.layer_params() * 4ull) / (2ull * d));
    r.ctx_max = std::max(256u, std::min(ctx_max_mem, ctx_parity));
    r.threads_used = max_threads;
    if (ctx_max_mem < ctx_parity) {
        r.check.notes.push_back(
            "可用内存不足以把上下文推到与权重等量的量级，解码段偏向权重受限；"
            "可减小模型规模或线程数。");
    }

    const uint32_t repeats = std::max(2u, opt_.repeats);
    const uint32_t max_repeats = std::max(repeats + 15u, 18u);
    r.check.min_repeats = repeats;

    // ---- weights: generated once, shared by every kernel in the run --------
    report("生成确定性权重", 0.05);
    WeightStorePtr w32 = make_weights(r.model, Precision::FP32);
    WeightStorePtr w8;
    if (opt_.with_int8) w8 = make_weights(r.model, Precision::INT8);

    // ---- self-check: the dispatched kernel must agree with the reference ----
    r.check.dispatch_err = 0.0;
    {
        Kernel k(w32, max_threads, kMLadderMax);
        r.check.dispatch_err = k.verify_dispatch(1e-3);
        if (r.check.dispatch_err > 0.0) {
            r.check.notes.push_back("AVX2 内核与标量参考内核结果不一致，已判定为无效测量。");
        }
        // Warm-up pass: faults every buffer in and settles the predictor. Kept
        // out of every recorded sample.
        report("预热", 0.10);
        k.run_prefill();
        k.prepare_decode(r.ctx_max);
        k.run_decode();
        r.check.bad_values = k.bad_values();
    }

    const auto ms = m_ladder();
    r.prefill_m.label = "算力随批量的变化（单线程）";
    r.prefill_m.x_label = "每线程 token 数 M";
    r.prefill_m.y_label = "GFLOP/s";

    for (size_t i = 0; i < ms.size(); ++i) {
        if (stop()) { r.cancelled = true; return r; }
        report("算力标定 M=" + std::to_string(ms[i]), 0.10 + 0.30 * (double)i / ms.size());
        Kernel k(w32, 1, ms[i]);
        k.run_prefill();  // warm-up, discarded
        const uint32_t inner = inner_reps_for(k.run_prefill(), kSampleSeconds);
        Stat st = sample_adaptive(
            [&]() {
                const double t0 = now_s();
                for (uint32_t j = 0; j < inner; ++j) k.run_prefill();
                return k.prefill_flops() * (double)inner / (now_s() - t0) / 1e9;
            },
            repeats, max_repeats, kTargetNoise, 4.0);
        r.check.worst_noise = std::max(r.check.worst_noise, st.rel_noise);
        r.prefill_m.x.push_back(ms[i]);
        r.prefill_m.y.push_back(st.median);
        r.prefill_m.err.push_back(st.rel_noise);
        if (ms[i] == 1) r.peak_gflops_1t = st.median;
        if (st.median > r.peak_gflops) {
            r.peak_gflops = st.median;
            r.peak_tokens = ms[i];
        }
    }

    // Implied single-thread bandwidth from the M=1 pass, for the self-check.
    {
        Kernel k(w32, 1, 1);
        k.run_prefill();
        std::vector<double> bw;
        for (uint32_t t = 0; t < repeats; ++t) {
            bw.push_back((double)r.model.weight_bytes(Precision::FP32) / k.run_prefill() / 1e9);
        }
        r.check.kernel_gb_s = summarize(bw).median;
    }
    r.check.bandwidth_ratio = r.stream_gb_s > 0.0 ? r.check.kernel_gb_s / r.stream_gb_s : 0.0;
    r.check.bandwidth_plausible =
        r.check.bandwidth_ratio > 0.20 && r.check.bandwidth_ratio < 1.35;
    if (!r.check.bandwidth_plausible) {
        char buf[192];
        snprintf(buf, sizeof(buf),
                 "内核实测带宽 %.2f GB/s 与访存参考 %.2f GB/s 相差 %.2f 倍，超出可信区间；"
                 "测量反映的是内核实现而非 CPU 能力。",
                 r.check.kernel_gb_s, r.stream_gb_s, r.check.bandwidth_ratio);
        r.check.notes.push_back(buf);
    }
    if (r.check.bad_values > 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "被测内核产生了 %llu 个 NaN/Inf，结果不可信。",
                 (unsigned long long)r.check.bad_values);
        r.check.notes.push_back(buf);
    }

    // ---- thread scaling ----------------------------------------------------
    r.thread_scaling.label = "多线程扩展性";
    r.thread_scaling.x_label = "线程数";
    r.thread_scaling.y_label = "tokens/s";

    const auto tl = thread_ladder(max_threads);
    for (size_t i = 0; i < tl.size(); ++i) {
        if (stop()) { r.cancelled = true; return r; }
        const uint32_t n = tl[i];
        report("线程扩展 n=" + std::to_string(n), 0.42 + 0.12 * (double)i / tl.size());
        Kernel k(w32, n, r.peak_tokens ? r.peak_tokens : 1);
        k.run_prefill();
        const uint32_t inner = inner_reps_for(k.run_prefill(), kSampleSeconds);
        Stat st = sample_adaptive(
            [&]() {
                const double t0 = now_s();
                for (uint32_t j = 0; j < inner; ++j) k.run_prefill();
                return (double)n * (r.peak_tokens ? r.peak_tokens : 1) * (double)inner /
                       (now_s() - t0);
            },
            repeats, max_repeats, kTargetNoise, 5.0);
        r.check.worst_noise = std::max(r.check.worst_noise, st.rel_noise);
        r.thread_scaling.x.push_back(n);
        r.thread_scaling.y.push_back(st.median);
        r.thread_scaling.err.push_back(st.rel_noise);
    }

    // ---- decode: context decay --------------------------------------------
    r.ctx_decay.label = "解码延迟随上下文增长";
    r.ctx_decay.x_label = "上下文长度";
    r.ctx_decay.y_label = "ms/token（单流）";

    const auto ladder = ctx_ladder(r.ctx_max, 4);
    std::vector<uint32_t> cl = ladder;
    std::vector<double> fp32_decode_s;

    // Memory can be reclaimed by other programs between the budget check and
    // the decode sweep -- this is by far the largest allocation the run makes.
    // Re-check and shorten the ladder rather than dying at the biggest step.
    const uint64_t kv_total = 2ull * d * 4ull * max_threads;
    {
        const MemoryInfo now_mem = detect_memory();
        uint64_t need = (uint64_t)cl.back() * kv_total;
        if (now_mem.avail_bytes && need > (uint64_t)((double)now_mem.avail_bytes * 0.55)) {
            while (cl.size() > 2 &&
                   (uint64_t)cl.back() * kv_total >
                       (uint64_t)((double)now_mem.avail_bytes * 0.55)) {
                cl.pop_back();
            }
            r.check.notes.push_back(
                "解码前可用内存下降，已缩短上下文阶梯以避免内存耗尽。");
        }
    }

    // One decode kernel for the whole ladder: the context only ever grows, so
    // the per-thread KV caches are allocated and first-touched exactly once.
    {
        Kernel k(w32, max_threads, 1);
        for (size_t i = 0; i < cl.size(); ++i) {
            if (stop()) { r.cancelled = true; return r; }
            const uint32_t c = cl[i];
            report("解码 ctx=" + std::to_string(c), 0.56 + 0.20 * (double)i / cl.size());
            k.prepare_decode(c);
            k.run_decode();
            const uint32_t inner = inner_reps_for(k.run_decode(), kSampleSeconds);
            Stat st = sample_adaptive(
                [&]() {
                    const double t0 = now_s();
                    for (uint32_t j = 0; j < inner; ++j) k.run_decode();
                    return (now_s() - t0) * 1000.0 / (double)inner;
                },
                repeats, max_repeats, kTargetNoise, 8.0);
            r.check.worst_noise = std::max(r.check.worst_noise, st.rel_noise);
            r.ctx_decay.x.push_back(c);
            r.ctx_decay.y.push_back(st.median);
            r.ctx_decay.err.push_back(st.rel_noise);
            fp32_decode_s.push_back(st.median);
        }
    }

    // ---- quantisation ------------------------------------------------------
    if (opt_.with_int8) {
        r.quant.label = "INT8 相对 FP32 的加速";
        r.quant.x_label = "上下文长度";
        r.quant.y_label = "加速比";
        Kernel k(w8, max_threads, 1);
        for (size_t i = 0; i < cl.size(); ++i) {
            if (stop()) { r.cancelled = true; return r; }
            const uint32_t c = cl[i];
            report("量化对比 ctx=" + std::to_string(c), 0.78 + 0.18 * (double)i / cl.size());
            k.prepare_decode(c);
            k.run_decode();
            const uint32_t inner = inner_reps_for(k.run_decode(), kSampleSeconds);
            Stat st = sample_adaptive(
                [&]() {
                    const double t0 = now_s();
                    for (uint32_t j = 0; j < inner; ++j) k.run_decode();
                    return (now_s() - t0) * 1000.0 / (double)inner;
                },
                repeats, max_repeats, kTargetNoise, 8.0);
            r.check.worst_noise = std::max(r.check.worst_noise, st.rel_noise);
            double speedup = fp32_decode_s[i] > 0.0 ? fp32_decode_s[i] / st.median : 0.0;
            r.quant.x.push_back(c);
            r.quant.y.push_back(speedup);
            r.quant.err.push_back(st.rel_noise);
        }
    }

    // `pass` covers only the conditions that make the measurement meaningless.
    // Noise is reported separately and handled by the scorer as a discount,
    // not as a veto.
    r.check.pass = r.check.bandwidth_plausible && r.check.dispatch_err == 0.0 &&
                   r.check.bad_values == 0;
    if (r.check.worst_noise >= 0.10) {
        char buf[128];
        snprintf(buf, sizeof(buf), "部分采样点的相对离散度最高 %.1f%%。",
                 r.check.worst_noise * 100.0);
        r.check.notes.push_back(buf);
    }
    r.total_seconds = now_s() - t_start;
    return r;
}

} // namespace icemark