#include "cpu_detector.hpp"
#include "weight_generator.hpp"
#include "inference_engine.hpp"
#include "benchmark.hpp"
#include "gui_chart.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

struct Config {
    std::string model_size = "small";
    std::string precision = "fp32";
};

ModelConfig get_model_config(const std::string& size) {
    if (size == "large") {
        return {4096, 32, 16384, 4096};
    } else if (size == "medium") {
        return {3072, 24, 12288, 4096};
    }
    return {2048, 16, 8192, 4096};
}

void print_usage(const char* prog) {
    std::cout << "IceMark - CPU AI Inference Benchmark\n"
              << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model-size small|medium|large  (default: small)\n"
              << "  --precision fp32|int8|both       (default: fp32)\n"
              << "  --help                          Show this message\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (strcmp(argv[i], "--model-size") == 0 && i + 1 < argc) {
            cfg.model_size = argv[++i];
        } else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            cfg.precision = argv[++i];
        }
    }
    return cfg;
}

uint64_t get_available_memory_bytes() {
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        return ms.ullTotalPhys;
    }
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        return (uint64_t)pages * (uint64_t)page_size;
    }
#endif
    return 0;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    std::cout << "======================================\n";
    std::cout << "  IceMark - CPU AI Inference Benchmark\n";
    std::cout << "======================================\n\n";

    CPUInfo cpu = detect_cpu();
    std::cout << "CPU Detection:\n";
    std::cout << "  Logical cores: " << cpu.logical_cores << "\n";
    std::cout << "  Physical cores: " << cpu.physical_cores << "\n";
    std::cout << "  SIMD: " << simd_capabilities_string(cpu) << "\n";
    std::cout << "\n";

    ModelConfig model = get_model_config(cfg.model_size);
    std::cout << "Model Configuration (" << cfg.model_size << "):\n";
    std::cout << "  dim=" << model.dim << ", layers=" << model.n_layers
              << ", hidden_dim=" << model.hidden_dim << "\n";

    uint64_t fp32_bytes = WeightGenerator::required_fp32_bytes(model);
    uint64_t int8_bytes = WeightGenerator::required_int8_bytes(model);
    std::cout << "  FP32 weight size: " << (fp32_bytes / (1024*1024)) << " MB\n";
    std::cout << "  INT8 weight size: " << (int8_bytes / (1024*1024)) << " MB\n";

    uint64_t total_mem = get_available_memory_bytes();
    if (total_mem > 0) {
        std::cout << "  System RAM: " << (total_mem / (1024*1024*1024)) << " GB\n";
    }

    uint64_t needed_fp32 = fp32_bytes + 2ULL * model.dim * model.context_len * model.n_layers * sizeof(float);
    uint64_t needed_int8 = int8_bytes + 2ULL * model.dim * model.context_len * model.n_layers * sizeof(float);

    uint32_t effective_threads = cpu.logical_cores;
    if (total_mem > 0) {
        uint64_t mem_needed = needed_fp32;
        if (cfg.precision == "both") mem_needed += needed_int8;
        uint64_t mem_per_thread = 2ULL * model.dim * model.context_len * model.n_layers * sizeof(float);
        uint64_t max_threads_mem = cpu.logical_cores * mem_per_thread;
        uint64_t mem_limit = (uint64_t)((double)total_mem * 0.8);
        if (mem_needed + max_threads_mem > mem_limit) {
            double mem_avail = ((double)total_mem * 0.8 - (double)fp32_bytes);
            double per_thread_mem = (double)mem_per_thread;
            effective_threads = (uint32_t)(mem_avail / per_thread_mem);
            if (effective_threads > cpu.logical_cores) effective_threads = cpu.logical_cores;
            if (effective_threads < 1) effective_threads = 1;
            std::cout << "\n  WARNING: Memory constrained. Limiting threads to " << effective_threads << "\n";
        }
    }

    std::cout << "\n";

    std::vector<uint32_t> thread_counts;
    for (uint32_t t = 1; t <= effective_threads; ++t) {
        thread_counts.push_back(t);
    }

    if (cfg.precision == "fp32" || cfg.precision == "both") {
        std::cout << "======================================\n";
        std::cout << "  FP32 Benchmark\n";
        std::cout << "======================================\n\n";

        InferenceEngine engine(model, Precision::FP32);

        std::vector<double> prefill_toks;
        std::vector<double> prefill_threads;

        std::cout << "Running Prefill benchmark (128-token sequence)...\n\n";
        for (uint32_t n : thread_counts) {
            engine.run_prefill(n);
            InferenceStats s = engine.get_last_stats();
            prefill_threads.push_back((double)n);
            prefill_toks.push_back(s.tokens_per_sec);
            std::cout << "  " << n << " thread(s): " << std::fixed << std::setprecision(1)
                      << s.tokens_per_sec << " tokens/s\n";
        }

        show_chart_window("FP32 Prefill (tokens/s)", "Threads", "tokens/s",
                         prefill_threads, prefill_toks);

        std::vector<double> context_lengths = {128, 512, 1024, 4096};
        for (double ctx : context_lengths) {
            std::vector<double> decode_ms;
            std::vector<double> decode_threads;

            std::cout << "\nRunning Decode benchmark (context_len=" << (int)ctx << ")...\n\n";
            for (uint32_t n : thread_counts) {
                engine.run_decode(n, (uint32_t)ctx);
                InferenceStats s = engine.get_last_stats();
                decode_threads.push_back((double)n);
                decode_ms.push_back(s.ms_token);
                std::cout << "  " << n << " thread(s): " << std::fixed << std::setprecision(2)
                          << s.ms_token << " ms/token\n";
            }

            std::ostringstream title;
            title << "FP32 Decode (ms/token, ctx=" << (int)ctx << ")";
            show_chart_window(title.str(), "Threads", "ms/token",
                             decode_threads, decode_ms);
        }
    }

    if (cfg.precision == "int8" || cfg.precision == "both") {
        std::cout << "\n======================================\n";
        std::cout << "  INT8 Benchmark\n";
        std::cout << "======================================\n\n";

        InferenceEngine engine(model, Precision::INT8);

        std::vector<double> prefill_toks;
        std::vector<double> prefill_threads;

        std::cout << "Running Prefill benchmark (128-token sequence)...\n\n";
        for (uint32_t n : thread_counts) {
            engine.run_prefill(n);
            InferenceStats s = engine.get_last_stats();
            prefill_threads.push_back((double)n);
            prefill_toks.push_back(s.tokens_per_sec);
            std::cout << "  " << n << " thread(s): " << std::fixed << std::setprecision(1)
                      << s.tokens_per_sec << " tokens/s\n";
        }

        show_chart_window("INT8 Prefill (tokens/s)", "Threads", "tokens/s",
                          prefill_threads, prefill_toks);
    }

    std::cout << "\n======================================\n";
    std::cout << "  Benchmark Complete\n";
    std::cout << "======================================\n";
    std::cout << "\nModel: " << cfg.model_size
              << " | Precision: " << cfg.precision
              << " | Threads: " << effective_threads << "\n";

    return 0;
}