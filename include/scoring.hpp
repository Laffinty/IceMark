#pragma once
#include "harness.hpp"

#include <string>
#include <vector>

namespace icemark {

// A log-anchored bracket. Linear anchoring would compress every real CPU into
// the bottom 10% of the scale, because capability spans decades while a linear
// map only spans the range you typed in.
struct Anchor {
    const char* label;
    double lo;             // score 0
    double hi;             // score 100
    bool measured;         // false = architecture-derived estimate
};

// Result of the pre-score self-check. When this fails the tool shows a reason
// instead of a number -- a score that survived an implausible measurement is
// worse than no score, because it looks precise.
struct Verdict {
    bool valid = false;
    double confidence = 0.0;  // 0..1, shown as a badge
    const char* grade = "";   // 可信 / 参考 / 不可信
    std::vector<std::string> reasons;
};

struct SubScore {
    std::string name;
    std::string detail;   // the measured quantity, formatted
    double value = 0.0;   // raw measurement
    double score = 0.0;   // 0..100
    double weight = 0.0;
};

struct Score {
    bool valid = false;
    double total = 0.0;
    Verdict verdict;
    std::vector<SubScore> parts;
};

const std::vector<Anchor>& scalar_gflops_anchors();
const std::vector<Anchor>& memory_bandwidth_anchors();

Verdict judge(const Result& r);
Score compute_score(const Result& r);

std::string format_gflops(double v);
std::string format_gbs(double v);
std::string format_ms(double v);

} // namespace icemark