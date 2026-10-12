#pragma once
#include "cpu.hpp"
#include "kernel.hpp"
#include "model.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace icemark {

// Median + MAD over `repeats` samples. A benchmark that reports a single
// sample has no error bar at all, which on this machine alone was worth 7%
// run-to-run drift -- larger than most of the differences being charted.
struct Stat {
    double median = 0.0;
    double mad = 0.0;        // median absolute deviation
    double rel_noise = 0.0;  // 1.4826*MAD / median
    double lo = 0.0, hi = 0.0;
    int n = 0;
    int rejected = 0;
};

Stat summarize(std::vector<double> samples);

struct Series {
    std::string label;
    std::string x_label;
    std::string y_label;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> err;  // relative noise, for error bars
};

// Preconditions that must hold before any number is allowed to become a
// score. These are hard gates, not warnings: a failure here means the
// measurement is reporting something other than the CPU's capability.
struct SelfCheck {
    double stream_gb_s = 0.0;      // reference sequential read bandwidth
    double kernel_gb_s = 0.0;      // bandwidth implied by the 1-thread kernel
    double bandwidth_ratio = 0.0;  // kernel / stream reference
    double dispatch_err = 0.0;     // AVX2 vs reference kernel disagreement
    uint64_t bad_values = 0;       // NaN/Inf seen inside the measured passes
    uint32_t min_repeats = 0;
    double worst_noise = 0.0;      // worst relative MAD across all samples

    bool bandwidth_plausible = false;
    bool pass = false;
    std::vector<std::string> notes;
};

struct RunOptions {
    Profile profile = Profile::Standard;
    bool with_int8 = true;
    uint32_t repeats = 3;
    uint32_t max_threads = 0;  // 0 = auto
};

struct Result {
    CPUInfo cpu;
    MemoryInfo mem;
    ModelConfig model;
    RunOptions opt;

    uint32_t threads_used = 0;
    uint32_t ctx_max = 0;
    const char* dot_kind = "";
    SelfCheck check;

    double stream_gb_s = 0.0;
    double peak_gflops = 0.0;
    uint32_t peak_tokens = 0;
    double peak_gflops_1t = 0.0;

    Series prefill_m;       // x = M tokens, y = GFLOP/s, 1 thread
    Series thread_scaling;  // x = threads, y = tokens/s
    Series ctx_decay;       // x = ctx, y = ms/token per stream
    Series quant;           // x = ctx, y = INT8 speedup

    double total_seconds = 0.0;
    bool cancelled = false;
    // Set when the run could not complete (typically the machine ran out of
    // memory part-way through the decode sweep). A benchmark that allocates
    // several GB must survive that instead of taking the process down.
    std::string error;
};

class Harness {
public:
    explicit Harness(const RunOptions& opt) : opt_(opt) {}

    // Called as (stage text, fraction 0..1). Cheap; called between samples.
    void on_progress(std::function<void(const std::string&, double)> cb) {
        progress_ = std::move(cb);
    }

    void request_cancel() { cancel_.store(true); }
    bool cancelled() const { return cancel_.load(); }

    Result run();

private:
    Result run_impl();
    void report(const std::string& stage, double frac) {
        if (progress_) progress_(stage, frac);
    }
    bool stop() const { return cancel_.load(); }

    RunOptions opt_;
    std::function<void(const std::string&, double)> progress_;
    std::atomic<bool> cancel_{false};
};

// Reference sequential-read bandwidth, used as the self-check yardstick.
double measure_stream_bandwidth_gb_s(size_t buffer_bytes);

} // namespace icemark