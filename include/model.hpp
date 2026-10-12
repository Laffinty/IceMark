#pragma once
#include <cstdint>

namespace icemark {

enum class Precision { FP32, INT8 };

// Per-layer parameter layout. Every matrix is stored row-major as
// [out_rows][in_cols] and every hot loop walks a single row contiguously:
//
//   Wq, Wk, Wv, Wo : dim x dim       -> 4 * dim^2
//   Wgate, Wup    : hidden x dim     -> 2 * dim * hidden
//   Wdown         : dim x hidden     -> 1 * dim * hidden
//
// Context length is deliberately NOT a member: it is a swept test parameter,
// not a property of the model. Keeping it here previously made the memory
// budget code average in a context that the benchmark never actually ran.
struct ModelConfig {
    uint32_t dim = 0;
    uint32_t n_layers = 0;
    uint32_t hidden_dim = 0;

    uint64_t layer_params() const { return 4ULL * dim * dim + 3ULL * dim * hidden_dim; }
    uint64_t total_params() const { return layer_params() * n_layers; }
    uint64_t weight_bytes(Precision p) const {
        return total_params() * (p == Precision::FP32 ? 4ull : 1ull);
    }
    // Bytes of weight actually touched by one forward pass. Every parameter is
    // live (Wo included), so this equals weight_bytes().
    uint64_t pass_bytes(Precision p) const { return weight_bytes(p); }

    bool valid() const { return dim != 0 && n_layers != 0 && hidden_dim != 0; }
};

enum class Profile { Quick, Standard, Deep };

const char* profile_name(Profile p);
ModelConfig profile_config(Profile p);

} // namespace icemark