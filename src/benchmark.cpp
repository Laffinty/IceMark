#include "benchmark.hpp"

Benchmark::Benchmark(uint32_t /*max_threads*/) {}
Benchmark::~Benchmark() {}

void Benchmark::add_prefill_result(uint32_t threads, double tokens_per_sec) {
    BenchmarkResult r;
    r.threads = threads;
    r.prefill_tokens_per_sec = tokens_per_sec;
    r.decode_ms_token = 0.0;
    r.decode_tokens_per_sec = 0.0;
    results_.push_back(r);
}

void Benchmark::add_decode_result(uint32_t threads, double ms_token, double tokens_per_sec) {
    BenchmarkResult r;
    r.threads = threads;
    r.prefill_tokens_per_sec = 0.0;
    r.decode_ms_token = ms_token;
    r.decode_tokens_per_sec = tokens_per_sec;
    results_.push_back(r);
}