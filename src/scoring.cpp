#include "scoring.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace icemark {
namespace {

// Maps v onto 0..100 across a log-spaced bracket.
double log_anchor(double v, double lo, double hi, bool* extrapolated) {
    if (!(v > 0.0) || !(lo > 0.0) || !(hi > lo)) return 0.0;
    const double t = (std::log2(v) - std::log2(lo)) / (std::log2(hi) - std::log2(lo));
    if (extrapolated) *extrapolated = (t < 0.0 || t > 1.0);
    return std::max(0.0, std::min(100.0, t * 100.0));
}

std::string fmt(const char* spec, double v) {
    char buf[96];
    snprintf(buf, sizeof(buf), spec, v);
    return buf;
}

std::string fmt2(const char* spec, double a, double b) {
    char buf[128];
    snprintf(buf, sizeof(buf), spec, a, b);
    return buf;
}

} // namespace

std::string format_gflops(double v) { return fmt("%.1f GFLOP/s", v); }
std::string format_gbs(double v) { return fmt("%.1f GB/s", v); }
std::string format_ms(double v) { return fmt("%.2f ms", v); }

// These brackets describe a SINGLE core, because that is what the harness
// measures. Anchoring a per-core number against a whole-machine ceiling (a
// 16-core part with 8 memory channels) is what made a healthy Zen2 score 10
// out of 100 on compute. The provenance flag is false because these are
// architecture-derived estimates, not measurements of other machines.
const std::vector<Anchor>& scalar_gflops_anchors() {
    static const std::vector<Anchor> a = {
        {"无 AVX2 的老核心", 8.0, 20.0, false},
        {"AVX2 桌面核心（低频）", 20.0, 60.0, false},
        {"AVX2 桌面核心（高频）", 60.0, 140.0, false},
    };
    return a;
}

const std::vector<Anchor>& memory_bandwidth_anchors() {
    static const std::vector<Anchor> a = {
        {"DDR3 / 单通道", 4.0, 10.0, false},
        {"DDR4 双通道", 10.0, 22.0, false},
        {"DDR5 / 服务器平台", 22.0, 45.0, false},
    };
    return a;
}

Verdict judge(const Result& r) {
    Verdict v;
    double conf = 1.0;

    if (r.cancelled) {
        v.valid = false;
        v.confidence = 0.0;
        v.grade = "不可信";
        v.reasons.push_back("测量被取消，结果不完整。");
        return v;
    }
    if (!r.error.empty()) {
        v.valid = false;
        v.confidence = 0.0;
        v.grade = "不可信";
        v.reasons.push_back(r.error);
        return v;
    }

    // Hard failures mean the numbers describe something other than the CPU, so
    // no score is emitted at all -- only a reason.
    const bool hard_fail = !r.check.bandwidth_plausible || r.check.dispatch_err > 0.0 ||
                           r.check.bad_values > 0;
    if (!r.check.bandwidth_plausible) {
        v.reasons.push_back(
            "内核实测带宽与访存参考相差过大，说明测到的是内核实现而不是 CPU 能力。");
    }
    if (r.check.dispatch_err > 0.0) {
        v.reasons.push_back("SIMD 内核与标量参考内核不一致。");
    }
    if (r.check.bad_values > 0) {
        v.reasons.push_back("被测内核产生了 NaN/Inf，数值路径已失效。");
    }

    // Soft problems discount confidence instead of vetoing the result. The
    // medians are already robust, so a noisy pass still says something true
    // about the machine -- refusing to say anything was strictly less useful
    // than saying it with a lower confidence.
    if (r.check.worst_noise >= 0.10) {
        v.reasons.push_back(fmt(
            "采样离散度最高 %.1f%%，结果已按噪声折算，建议增加重复次数后重跑。",
            r.check.worst_noise * 100.0));
        conf *= (r.check.worst_noise >= 0.35) ? 0.30
               : (r.check.worst_noise >= 0.20) ? 0.60
                                               : 0.85;
    }
    if (r.ctx_max < 1024) {
        v.reasons.push_back("可用内存不足以构造有意义的上下文长度，解码项按模型外推。");
        conf *= 0.9;
    }

    v.confidence = std::max(0.0, std::min(1.0, conf));
    v.valid = !hard_fail && v.confidence >= 0.4;
    if (!v.valid) v.grade = "不可信";
    else if (v.confidence >= 0.85) v.grade = "可信";
    else if (v.confidence >= 0.6) v.grade = "参考";
    else v.grade = "仅供参考";
    return v;
}

Score compute_score(const Result& r) {
    Score s;
    s.verdict = judge(r);

    const double w_bytes = (double)r.model.weight_bytes(Precision::FP32);
    const double d = (double)r.model.dim;
    const double L = (double)r.model.n_layers;

    // 1. Single-core FP32 capability, measured at the batch size where the
    //    workload finally clears the memory roof.
    {
        SubScore p;
        p.name = "单核算力（FP32）";
        p.value = r.peak_gflops;
        p.weight = 30.0;
        p.detail = format_gflops(r.peak_gflops) + "（M=" + std::to_string(r.peak_tokens) + "）";
        bool ex = false;
        p.score = log_anchor(r.peak_gflops, 8.0, 140.0, &ex);
        if (ex) p.detail += "（超出锚点区间）";
        s.parts.push_back(p);
    }

    // 2. Single-core streaming bandwidth.
    {
        SubScore p;
        p.name = "单核内存带宽";
        p.value = r.stream_gb_s;
        p.weight = 25.0;
        p.detail = format_gbs(r.stream_gb_s);
        bool ex = false;
        p.score = log_anchor(r.stream_gb_s, 4.0, 45.0, &ex);
        if (ex) p.detail += "（超出锚点区间）";
        s.parts.push_back(p);
    }

    // 3. Thread scaling efficiency, measured at full thread count.
    //
    //    Two earlier behaviours were wrong. Taking the PEAK efficiency over
    //    all thread counts let n=2 win with a 105% superlinear reading that was
    //    purely the +-8% noise on the single-thread baseline, and it scored a
    //    flat 100 on every machine. The n=1 point also divides by itself and is
    //    therefore always exactly 1.0.
    {
        SubScore p;
        p.name = "多线程扩展性";
        p.weight = 20.0;
        double base = 0.0;
        for (size_t i = 0; i < r.thread_scaling.x.size(); ++i) {
            if (r.thread_scaling.x[i] == 1.0) base = r.thread_scaling.y[i];
        }
        double peak = 0.0;
        uint32_t peak_at = 0;
        double full = 0.0;
        uint32_t full_at = 0;
        for (size_t i = 0; i < r.thread_scaling.x.size(); ++i) {
            const double n = r.thread_scaling.x[i];
            if (n <= 1.0 || base <= 0.0) continue;
            const double eff = r.thread_scaling.y[i] / (base * n);
            if (eff > peak) {
                peak = eff;
                peak_at = (uint32_t)n;
            }
            full = eff;
            full_at = (uint32_t)n;
        }
        p.value = full;
        char b2[128];
        snprintf(b2, sizeof(b2), "满载效率 %.0f%%（%u 线程），峰值 %.0f%%（%u 线程）",
                 full * 100.0, full_at, peak * 100.0, peak_at);
        p.detail = b2;
        // A bandwidth-bound workload saturates well before 100% linear, so the
        // 50%..100% mapping handed healthy machines a zero. Full marks sit at
        // 85%, which is close to what a well-behaved SMT box actually reaches.
        p.score = std::max(0.0, std::min(100.0, (full - 0.25) / 0.60 * 100.0));
        s.parts.push_back(p);
    }

    // 4. Context decay, scored against the Roofline prediction from the byte
    //    model rather than against an arbitrary curve.
    {
        SubScore p;
        p.name = "上下文衰减一致性";
        p.weight = 15.0;
        if (r.ctx_decay.y.size() >= 2) {
            const double c0 = r.ctx_decay.x.front();
            const double c1 = r.ctx_decay.x.back();
            const double measured = r.ctx_decay.y.back() / std::max(1e-9, r.ctx_decay.y.front());
            const double kv0 = 2.0 * c0 * d * 4.0 * L;
            const double kv1 = 2.0 * c1 * d * 4.0 * L;
            const double predicted = (w_bytes + kv1) / std::max(1e-9, (w_bytes + kv0));
            p.value = measured;
            p.detail = fmt2("实测 ×%.2f，字节模型预测 ×%.2f", measured, predicted);
            if (predicted > 1.0 && measured > 1.0) {
                // The byte model is first order: it assumes every byte streams
                // at the same rate, but the KV cache is read as a gather over
                // `ctx` scattered rows, which costs extra bandwidth at large
                // contexts. A healthy machine therefore overshoots the model
                // by a few times; only a much larger gap means decode is doing
                // something the model does not describe at all.
                const double dev = std::fabs(std::log(measured) / std::log(predicted) - 1.0);
                p.score = dev <= 0.35 ? 100.0
                                    : std::max(0.0, 100.0 - (dev - 0.35) / 0.85 * 100.0);
            } else {
                p.score = 50.0;
            }
        } else {
            p.score = 0.0;
            p.detail = "数据不足";
        }
        s.parts.push_back(p);
    }

    // 5. Quantisation gain, scored per context against the byte-model ceiling.
    //    Only the weights shrink 4x, so the ceiling falls towards 1 as the KV
    //    cache comes to dominate -- INT8 genuinely cannot help there, and
    //    scoring it against a flat 4x would punish correct behaviour.
    {
        SubScore p;
        p.name = "量化收益（INT8）";
        p.weight = 10.0;
        if (!r.quant.y.empty()) {
            double acc = 0.0;
            double worst_ratio = std::numeric_limits<double>::infinity();
            int counted = 0;
            for (size_t i = 0; i < r.quant.y.size(); ++i) {
                const double c = r.quant.x[i];
                const double kv = 2.0 * c * d * 4.0 * L;
                const double ceiling = (w_bytes + kv) / std::max(1e-9, (w_bytes * 0.25 + kv));
                if (ceiling <= 1.0) continue;
                acc += std::min(1.0, r.quant.y[i] / ceiling);
                worst_ratio = std::min(worst_ratio, r.quant.y[i]);
                ++counted;
            }
            p.value = std::isfinite(worst_ratio) ? worst_ratio : 0.0;
            if (counted > 0) {
                p.score = acc / counted * 100.0;
                p.detail = fmt2("平均达成 %.0f%%，最大 ctx 加速 ×%.2f",
                                p.score, std::isfinite(worst_ratio) ? worst_ratio : 0.0);
            } else {
                p.score = 0.0;
                p.detail = "未测量";
            }
        } else {
            p.score = 0.0;
            p.detail = "未测量";
        }
        s.parts.push_back(p);
    }

    double wsum = 0.0, acc = 0.0;
    for (const auto& p : s.parts) {
        wsum += p.weight;
        acc += p.score * p.weight;
    }
    s.total = wsum > 0.0 ? acc / wsum : 0.0;

    if (!s.verdict.valid) {
        s.valid = false;
    } else {
        s.valid = true;
        // Score and confidence are two independent signals and are reported as
        // such. Folding confidence into the headline number made two runs of
        // the same machine differ by 5 points purely because one sampling point
        // happened to be noisy, while every sub-score agreed to within 1 point.
        // Confidence is shown as its own badge; here it only trims the edges.
        s.total *= (0.85 + 0.15 * s.verdict.confidence);
    }
    return s;
}

} // namespace icemark