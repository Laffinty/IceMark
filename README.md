# IceMark

A lightweight CPU inference benchmark tool. Measures token/s across different thread counts, simulates transformer inference with KV-Cache and memory-bandwidth pressure.

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_CXX_STANDARD=17
cmake --build . --config Release

# Run (FP32, small model, ~1GB RAM required)
./build/Release/icemark.exe --model-size small --precision fp32

# INT8 mode
./build/Release/icemark.exe --model-size small --precision int8

# Full test (FP32 + INT8)
./build/Release/icemark.exe --model-size small --precision both
```

## Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--model-size` | `small` / `medium` / `large` | `small` | Model dimension and layer count |
| `--precision` | `fp32` / `int8` / `both` | `fp32` | Weight data type |
| `--help` | — | — | Show usage |

### Model Sizes

| Config | dim | Layers | FP32 | Min RAM | Target |
|--------|-----|--------|------|---------|--------|
| `small` | 2048 | 16 | ~1 GB | 8 GB | General / laptop |
| `medium` | 3072 | 24 | ~3 GB | 16 GB | Desktop |
| `large` | 4096 | 32 | ~8 GB | 32 GB | Workstation |

## Theory of Operation

### Algorithm

IceMark simulates a transformer layer forward pass with three sequential stages:

**Stage 1 — QKV Projection (Prefill)**
For a sequence of `L` tokens with model dimension `d`:

```
Q = X · W_q ∈ ℝ^{L×d}
K = X · W_k ∈ ℝ^{L×d}
V = X · W_v ∈ ℝ^{L×d}
```

Each projection is a **GEMM** (matrix × matrix). Computation: `O(L · d²)` per projection, `3·L·d²` total for Q/K/V.

**Stage 2 — Attention with KV-Cache (Decode)**
At step `t` with context length `c`, the model attends over all previous tokens:

```
score_i = (q · k_i) / √d   for i = 1..c
a_i = softmax(score)_i
output = Σ(a_i · v_i)
```

Each step performs `c` **GEMV** operations (matrix × vector) reading the full KV-Cache. Computation: `O(c·d)` per step. As `c` grows to 4096+, performance decays due to memory bandwidth saturation — this is the real bottleneck in CPU inference.

**Stage 3 — FFN**
```
gate = σ(X · W_gate)          ∈ ℝ^{d}
up   = tanh(X · W_up)        ∈ ℝ^{d}
out  = gate · up · W_down     ∈ ℝ^{d}
```

Each FFN step performs 3 GEMVs. Computation: `O(3·d²)` per token.

### Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Prefill tokens/s** | `128 / T_prefill` | Throughput of processing a full prompt (128 tokens) |
| **Decode ms/token** | `T_decode` | Latency of generating one token (lower is better) |
| **INT8 speedup** | `T_fp32 / T_int8` | Throughput improvement vs FP32 (memory bandwidth advantage) |

### Memory Layout

- **Weights**: shared read-only across all threads (not duplicated per thread)
- **KV-Cache**: per-thread private buffer, grows with context length
- **Activations**: per-thread private, O(d) floats

Weight matrix total: `4·d² + 3·d·d_hidden` per layer (Q/K/V + FFN). Total model: `O(4·d²·n_layers)` parameters. All weights generated deterministically via LCG (seed=42) for reproducibility.

## Project Structure

```
IceMark/
├── src/
│   ├── main.cpp              # Entry, CLI, benchmark dispatch
│   ├── cpu_detector.cpp      # Logical cores + SIMD detection
│   ├── weight_generator.cpp  # LCG pseudo-random weight generation
│   ├── inference_engine.cpp  # Transformer layer simulation
│   ├── benchmark.cpp         # Result recording
│   └── renderer.cpp          # ASCII line chart rendering
├── include/                  # Headers
├── docs/CONSTRUCTION.md     # Full technical specification
└── CMakeLists.txt
```

## Output Example

```
======================================
  IceMark - CPU AI Inference Benchmark
======================================

CPU Detection:
  Logical cores: 16
  Physical cores: 8
  SIMD: AVX2 AVX-512

Model Configuration (small):
  dim=2048, layers=16, hidden_dim=8192
  FP32 weight size: 1024 MB
  INT8 weight size: 256 MB

======================================
  FP32 Benchmark
======================================

Running Prefill benchmark (128-token sequence)...

  1 thread(s): 1423.1 tokens/s
  2 thread(s): 2780.4 tokens/s
  ...

  Prefill (tokens/s)
           3000 |          
               |        **
           2250 |      **  **
               |    **      **
           1500 |  **          **
               |**              **
               +------------------
                     1  2  3 ...
```

## Build Requirements

- C++17 compiler (MSVC 2019+, GCC 9+, Clang 12+)
- CMake ≥ 3.15
- No external dependencies (standard library only)

## License

GPLv2
