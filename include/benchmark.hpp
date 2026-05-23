#pragma once
#include <cstdint>
#include <vector>

struct BenchmarkResult {
    uint32_t threads;
    double prefill_tokens_per_sec;
    double decode_ms_token;
    double decode_tokens_per_sec;
};

class Benchmark {
public:
    Benchmark(uint32_t max_threads);
    ~Benchmark();

    void add_prefill_result(uint32_t threads, double tokens_per_sec);
    void add_decode_result(uint32_t threads, double ms_token, double tokens_per_sec);

    std::vector<BenchmarkResult> get_results() const { return results_; }
    void clear_results() { results_.clear(); }

private:
    std::vector<BenchmarkResult> results_;
};