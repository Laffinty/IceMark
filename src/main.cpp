#include "gui.hpp"
#include "scoring.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

using namespace icemark;

// ---------------------------------------------------------------------------
// Output plumbing. The binary is built as a GUI-subsystem app so that
// double-clicking it never flashes a console, so a headless run may find that
// there is no console to attach to at all (scheduled task, CI, service host).
// Everything is buffered and then flushed to whichever sink is available.
// ---------------------------------------------------------------------------

static std::string g_report;

static void emit(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_report += buf;
}

static bool enable_console() {
#if defined(_WIN32) || defined(_WIN64)
    const bool ok = (AttachConsole(ATTACH_PARENT_PROCESS) != 0) || (AllocConsole() != 0);
    if (ok) {
        FILE* f = nullptr;
        freopen_s(&f, "CONOUT$", "w", stdout);
        freopen_s(&f, "CONOUT$", "w", stderr);
    }
    SetConsoleOutputCP(CP_UTF8);
    return ok;
#else
    return true;
#endif
}

static void flush_report(bool have_console, const std::string& out_path) {
    if (!out_path.empty()) {
        FILE* fp = nullptr;
        if (fopen_s(&fp, out_path.c_str(), "wb") == 0 && fp) {
            std::fwrite(g_report.data(), 1, g_report.size(), fp);
            // UTF-8 BOM so Notepad opens Chinese text correctly.
            const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
            std::fseek(fp, 0, SEEK_SET);
            std::fwrite(bom, 1, 3, fp);
            std::fclose(fp);
            if (have_console) emit("报告已写入: %s\n", out_path.c_str());
        } else if (have_console) {
            emit("报告写入失败: %s\n", out_path.c_str());
        }
    }
    if (have_console) std::fwrite(g_report.data(), 1, g_report.size(), stdout);
}

static void print_usage() {
    emit(
        "IceMark 2.0.0 - CPU 算力测评\n"
        "用法: icemark [选项]\n"
        "  --cli                 强制命令行模式（不启动 GUI）\n"
        "  --out <文件>          把报告写入 UTF-8 文件\n"
        "  --profile quick|standard|deep   模型规模（默认 standard）\n"
        "  --repeats N           每个采样点重复次数（默认 3，下限 3）\n"
        "  --threads N           线程上限（默认=逻辑核数）\n"
        "  --no-int8             跳过 INT8 对比\n"
        "  --help                显示本帮助\n");
}

static int run_headless(const RunOptions& opt, bool have_console, const std::string& out_path) {
    Harness h(opt);
    h.on_progress([](const std::string& stage, double frac) {
        emit("  [%3d%%] %s\n", (int)(frac * 100.0), stage.c_str());
    });

    Result r = h.run();
    if (r.cancelled) {
        emit("\n已取消。\n");
        flush_report(have_console, out_path);
        return 2;
    }
    if (!r.error.empty()) {
        emit("\n测量未能完成：%s\n", r.error.c_str());
        flush_report(have_console, out_path);
        return 3;
    }

    Score s = compute_score(r);

    emit("\n================ 机器信息 ================\n");
    if (!r.cpu.brand.empty()) emit("  CPU      : %s\n", r.cpu.brand.c_str());
    emit("  核心     : %u 物理 / %u 逻辑  (%s)\n", r.cpu.physical_cores,
         r.cpu.logical_cores, r.cpu.has_ht ? "SMT" : "无 SMT");
    emit("  指令集   : %s  [内核: %s]\n", simd_capabilities_string(r.cpu).c_str(), r.dot_kind);
    emit("  内存     : %.1f GB 总 / %.1f GB 可用\n", r.mem.total_bytes / 1073741824.0,
         r.mem.avail_bytes / 1073741824.0);
    emit("  模型     : dim=%u layers=%u hidden=%u  权重 %.2f GB\n", r.model.dim,
         r.model.n_layers, r.model.hidden_dim,
         r.model.weight_bytes(Precision::FP32) / 1073741824.0);
    emit("  线程     : %u   最大上下文: %u\n", r.threads_used, r.ctx_max);

    emit("\n================ 自检 ================\n");
    emit("  访存参考     : %.2f GB/s\n", r.check.stream_gb_s);
    emit("  内核实测带宽 : %.2f GB/s  (比值 %.2fx)\n", r.check.kernel_gb_s,
         r.check.bandwidth_ratio);
    emit("  内核交叉校验 : %s\n", r.check.dispatch_err == 0.0 ? "通过" : "不一致");
    emit("  NaN/Inf 计数 : %llu\n", (unsigned long long)r.check.bad_values);
    emit("  最大采样噪声 : %.1f%%\n", r.check.worst_noise * 100.0);
    emit("  判定         : %s\n", r.check.pass ? "通过" : "不通过");

    emit("\n================ 评分 ================\n");
    if (!s.valid) {
        emit("  测量不可信，已拒绝出分。原因：\n");
        for (const auto& why : s.verdict.reasons) emit("    - %s\n", why.c_str());
        emit("  置信度 %.0f%%\n", s.verdict.confidence * 100.0);
    } else {
        emit("  总分 %.1f / 100      [置信度 %.0f%% · %s]\n\n", s.total,
             s.verdict.confidence * 100.0, s.verdict.grade);
        for (const auto& p : s.parts) {
            emit("    %-22s %5.1f 分  (权重 %2.0f%%)  %s\n", p.name.c_str(), p.score,
                 p.weight, p.detail.c_str());
        }
    }

    emit("\n算力标定 (单线程, x=M 每线程 token):\n");
    for (size_t i = 0; i < r.prefill_m.x.size(); ++i) {
        emit("    M=%-3u  %7.1f GFLOP/s  ±%.1f%%\n", (unsigned)r.prefill_m.x[i],
             r.prefill_m.y[i], r.prefill_m.err[i] * 100.0);
    }
    emit("\n线程扩展:\n");
    for (size_t i = 0; i < r.thread_scaling.x.size(); ++i) {
        emit("    %2u 线程  %9.1f tokens/s  ±%.1f%%\n", (unsigned)r.thread_scaling.x[i],
             r.thread_scaling.y[i], r.thread_scaling.err[i] * 100.0);
    }
    emit("\n上下文衰减 (单流延迟):\n");
    for (size_t i = 0; i < r.ctx_decay.x.size(); ++i) {
        emit("    ctx=%-7u %9.2f ms/token\n", (unsigned)r.ctx_decay.x[i], r.ctx_decay.y[i]);
    }
    if (!r.quant.y.empty()) {
        emit("\nINT8 加速比:\n");
        for (size_t i = 0; i < r.quant.x.size(); ++i) {
            emit("    ctx=%-7u  x%.2f\n", (unsigned)r.quant.x[i], r.quant.y[i]);
        }
    }
    emit("\n总耗时 %.1f 秒\n", r.total_seconds);
    flush_report(have_console, out_path);
    return 0;
}

static int app_main(int argc, char** argv) {
    RunOptions opt;
    bool cli = false;
    std::string out_path;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            const bool hc = enable_console();
            print_usage();
            flush_report(hc, out_path);
            return 0;
        } else if (strcmp(a, "--cli") == 0) {
            cli = true;
        } else if (strcmp(a, "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(a, "--no-int8") == 0) {
            opt.with_int8 = false;
        } else if (strcmp(a, "--profile") == 0 && i + 1 < argc) {
            const char* p = argv[++i];
            if (strcmp(p, "quick") == 0) opt.profile = Profile::Quick;
            else if (strcmp(p, "deep") == 0) opt.profile = Profile::Deep;
            else opt.profile = Profile::Standard;
        } else if (strcmp(a, "--repeats") == 0 && i + 1 < argc) {
            opt.repeats = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(a, "--threads") == 0 && i + 1 < argc) {
            opt.max_threads = (uint32_t)atoi(argv[++i]);
        }
    }

    // A score needs at least three samples before it can say anything about
    // its own noise.
    if (opt.repeats < 3) opt.repeats = 3;

    const char* no_gui = std::getenv("NO_GUI");
    if (no_gui && no_gui[0] == '1') cli = true;

    if (cli) {
        const bool hc = enable_console();
        return run_headless(opt, hc, out_path);
    }
    return run_gui(opt);
}

#if defined(_WIN32) || defined(_WIN64)

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    int argc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv) {
        const char* dummy[] = {"icemark"};
        return app_main(1, const_cast<char**>(dummy));
    }

    std::vector<std::string> store;
    std::vector<char*> argv_utf8;
    store.reserve((size_t)argc);
    argv_utf8.reserve((size_t)argc + 1);
    for (int i = 0; i < argc; ++i) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        store.emplace_back((size_t)n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &store.back()[0], n, nullptr, nullptr);
        argv_utf8.push_back(&store.back()[0]);
    }
    argv_utf8.push_back(nullptr);
    LocalFree(wargv);

    return app_main(argc, argv_utf8.data());
}

#endif