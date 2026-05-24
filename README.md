# IceMark

A lightweight, zero-dependency synthetic CPU benchmark that emulates the compute and memory-access patterns of transformer-based large language model (LLM) inference. It is designed for directional performance analysis—thread scaling, precision trade-offs, and context-length decay—rather than for absolute throughput prediction.

IceMark 是一个零外部依赖的轻量级 CPU 合成基准测试工具，其设计目标是复现基于 Transformer 的大型语言模型（LLM）推理过程中的计算与内存访问特征。该工具面向方向性性能分析——包括线程扩展性、精度模式权衡以及上下文长度衰减——而非绝对吞吐量的精确预测。

---

## 1. Computational Model

### 1.1 Workload Abstraction

IceMark simulates a decoder-only transformer stack [Vaswani et al., 2017] in two distinct inference regimes: **Prefill** (prompt processing) and **Decode** (autoregressive token generation). Each regime targets a different bound in the Roofline performance model [Williams et al., 2009]:

- **Prefill** is compute-bound. Each thread executes a full forward pass through all layers with a short input vector, exercising generalized matrix-vector multiplication (GEMV) over the stacked weight matrices.
- **Decode** is memory-bandwidth-bound. Each thread maintains a private KV-cache and, at every generation step, reads the entire historical key and value sequence back from DRAM. The resulting $O(c \cdot d)$ memory traffic per layer—where $c$ is the context length and $d$ is the model dimension—dominates over the $O(d^2)$ compute, pushing the working set out of CPU last-level cache and onto the memory bus.

IceMark 通过一个仅含解码器的 Transformer 堆栈 [Vaswani et al., 2017] 在两个 distinct 推理阶段中建立计算模型：**Prefill（提示处理）**与 **Decode（自回归 token 生成）**。两个阶段分别对应 Roofline 性能模型 [Williams et al., 2009] 中的不同约束区域：

- **Prefill** 受计算约束。每个线程以短输入向量执行全部层的完整前向传播，对堆叠的权重矩阵执行广义矩阵-向量乘法（GEMV）。
- **Decode** 受内存带宽约束。每个线程维护私有的 KV-Cache，并在每一步生成时从历史序列中读取全部键和值。每层产生的 $O(c \cdot d)$ 内存流量（$c$ 为上下文长度，$d$ 为模型维度）远超 $O(d^2)$ 的计算量，迫使工作集溢出 CPU 末级缓存并进入内存总线。

### 1.2 Layer Forward Pass (Formal Definition)

For a single layer indexed by $l$, let $\mathbf{x} \in \mathbb{R}^{d}$ denote the input activation vector. The per-layer forward pass is defined as follows:

**1. QKV Projection.** Three independent GEMV operations project the input into query, key, and value vectors:

$$
\mathbf{q} = \mathbf{x} \mathbf{W}_q^{(l)}, \quad
\mathbf{k} = \mathbf{x} \mathbf{W}_k^{(l)}, \quad
\mathbf{v} = \mathbf{x} \mathbf{W}_v^{(l)}
$$

where $\mathbf{W}_q^{(l)}, \mathbf{W}_k^{(l)}, \mathbf{W}_v^{(l)} \in \mathbb{R}^{d \times d}$. An output projection matrix $\mathbf{W}_o^{(l)}$ is allocated in the weight buffer but is **not exercised** in the current kernel, following the simplified attention formulation used in the implementation.

**2. KV-Cache Update.** If a KV-cache buffer $\mathbf{H}_{KV}$ is provided, the new key and value vectors are written into the slot indexed by the current decode step $t$:

$$
\mathbf{H}_{K}[t] \leftarrow \mathbf{k}, \quad \mathbf{H}_{V}[t] \leftarrow \mathbf{v}
$$

**3. Causal Self-Attention.** For context length $c$ (history length), attention scores are computed as scaled dot-products:

$$
\text{score}_i = \frac{\mathbf{q} \cdot \mathbf{H}_{K}[i]}{\sqrt{d}}, \quad i = 0, \ldots, c-1
$$

A numerically stable softmax is applied with max-value shifting:

$$
\alpha_i = \frac{\exp(\text{score}_i - \max_j \text{score}_j)}{\sum_{k} \exp(\text{score}_k - \max_j \text{score}_j)}
$$

The attention output is the weighted sum over the value cache:

$$
\mathbf{a} = \sum_{i=0}^{c-1} \alpha_i \, \mathbf{H}_{V}[i]
$$

When no cache is present (Prefill phase), the implementation bypasses the attention reduction and propagates $\mathbf{v}$ directly.

**4. Feed-Forward Network (FFN).** The FFN employs a gated linear unit (GLU) variant with sigmoid and hyperbolic-tangent gating:

$$
\mathbf{g} = \sigma(\mathbf{x} \mathbf{W}_{\text{gate}}^{(l)}), \quad
\mathbf{u} = \tanh(\mathbf{x} \mathbf{W}_{\text{up}}^{(l)})
$$

where $\sigma$ is the sigmoid function and $\tanh$ is approximated via the exponential form $\tanh(x) = (1 - e^{-2|x|}) / (1 + e^{-2|x|})$. The gated intermediate is projected back to the model dimension:

$$
\mathbf{f} = (\mathbf{g} \odot \mathbf{u}) \, {\mathbf{W}_{\text{down}}^{(l)}}^\top
$$

with $\mathbf{W}_{\text{gate}}^{(l)}, \mathbf{W}_{\text{up}}^{(l)} \in \mathbb{R}^{d \times h}$ and $\mathbf{W}_{\text{down}}^{(l)} \in \mathbb{R}^{h \times d}$. Here $h$ denotes the hidden dimension.

**5. Residual Addition.** The final layer output is an element-wise residual sum. Note that **layer normalization is omitted** in the current implementation, consistent with the scalar reference kernel:

$$
\mathbf{y} = \mathbf{a} + \mathbf{f}
$$

---

对于单层索引 $l$，设 $\mathbf{x} \in \mathbb{R}^{d}$ 为输入激活向量。单层的正式前向传播定义如下：

**1. QKV 投影。** 三个独立的 GEMV 操作将输入映射为查询、键和值向量：

$$
\mathbf{q} = \mathbf{x} \mathbf{W}_q^{(l)}, \quad
\mathbf{k} = \mathbf{x} \mathbf{W}_k^{(l)}, \quad
\mathbf{v} = \mathbf{x} \mathbf{W}_v^{(l)}
$$

其中 $\mathbf{W}_q^{(l)}, \mathbf{W}_k^{(l)}, \mathbf{W}_v^{(l)} \in \mathbb{R}^{d \times d}$。输出投影矩阵 $\mathbf{W}_o^{(l)}$ 在权重缓冲区中分配，但在当前内核中**未被使用**，这与实现中采用的简化注意力形式一致。

**2. KV-Cache 更新。** 若提供了 KV-Cache 缓冲区 $\mathbf{H}_{KV}$，则新的键和值向量被写入当前解码步 $t$ 对应的槽位：

$$
\mathbf{H}_{K}[t] \leftarrow \mathbf{k}, \quad \mathbf{H}_{V}[t] \leftarrow \mathbf{v}
$$

**3. 因果自注意力。** 对于上下文长度 $c$（历史长度），注意力分数通过缩放点积计算：

$$
\text{score}_i = \frac{\mathbf{q} \cdot \mathbf{H}_{K}[i]}{\sqrt{d}}, \quad i = 0, \ldots, c-1
$$

随后应用数值稳定的 softmax（采用最大值平移）：

$$
\alpha_i = \frac{\exp(\text{score}_i - \max_j \text{score}_j)}{\sum_{k} \exp(\text{score}_k - \max_j \text{score}_j)}
$$

注意力输出为值缓存的加权和：

$$
\mathbf{a} = \sum_{i=0}^{c-1} \alpha_i \, \mathbf{H}_{V}[i]
$$

当不存在缓存时（Prefill 阶段），实现跳过注意力归约，直接传播 $\mathbf{v}$。

**4. 前馈网络（FFN）。** FFN 采用带 sigmoid 与双曲正弦门控的门控线性单元（GLU）变体：

$$
\mathbf{g} = \sigma(\mathbf{x} \mathbf{W}_{\text{gate}}^{(l)}), \quad
\mathbf{u} = \tanh(\mathbf{x} \mathbf{W}_{\text{up}}^{(l)})
$$

其中 $\sigma$ 为 sigmoid 函数，$\tanh$ 通过指数形式近似：$\tanh(x) = (1 - e^{-2|x|}) / (1 + e^{-2|x|})$。门控中间结果被投影回模型维度：

$$
\mathbf{f} = (\mathbf{g} \odot \mathbf{u}) \, {\mathbf{W}_{\text{down}}^{(l)}}^\top
$$

其中 $\mathbf{W}_{\text{gate}}^{(l)}, \mathbf{W}_{\text{up}}^{(l)} \in \mathbb{R}^{d \times h}$，$\mathbf{W}_{\text{down}}^{(l)} \in \mathbb{R}^{h \times d}$，$h$ 为隐藏维度。

**5. 残差相加。** 最终层输出为逐元素的残差和。注意当前实现中**省略了层归一化（Layer Normalization）**，这与标量参考内核的设计保持一致：

$$
\mathbf{y} = \mathbf{a} + \mathbf{f}
$$

### 1.3 Complexity and Memory Footprint

Per-layer parameter count:

$$
P_{\text{layer}} = 4d^{2} + 3dh
$$

With $h = 4d$, this yields approximately $16d^{2}$ parameters per layer. The total weight memory is $P_{\text{layer}} \cdot L$ bytes for INT8 and $4 \cdot P_{\text{layer}} \cdot L$ bytes for FP32, where $L$ is the number of layers. The per-thread KV-cache consumes $2 \cdot c \cdot d \cdot 4$ bytes. For the default "small" configuration ($d=2048$, $L=16$, $c=4096$), the FP32 weight set is approximately 1 GB and the per-thread KV-cache is approximately 64 MB—both deliberately sized to exceed typical CPU last-level caches.

每层参数数量：

$$
P_{\text{layer}} = 4d^{2} + 3dh
$$

取 $h = 4d$，则每层约 $16d^{2}$ 个参数。总权重内存为 INT8 模式下 $P_{\text{layer}} \cdot L$ 字节，FP32 模式下 $4 \cdot P_{\text{layer}} \cdot L$ 字节，其中 $L$ 为层数。每线程 KV-Cache 占用 $2 \cdot c \cdot d \cdot 4$ 字节。以默认 "small" 配置为例（$d=2048$, $L=16$, $c=4096$），FP32 权重集约为 1 GB，每线程 KV-Cache 约为 64 MB——两者均被刻意设计为超出典型 CPU 末级缓存容量。

---

## 2. Deterministic Weight Generation

To guarantee bit-exact reproducibility across platforms and builds, weights are synthesized by a 64-bit Linear Congruential Generator (LCG) [Park & Miller, 1988; Knuth, 1997] with a fixed seed:

$$
x_{n+1} = a \cdot x_{n} + c \pmod{2^{64}}, \quad a = 6364136223846793005, \quad c = 1442695040888963407
$$

- **FP32 mode:** `float = (((state >> 33) / 16777216.0f) * 2.0f - 1.0f)` maps to $[-1, 1]$.
- **INT8 mode:** `int8_t = (int8_t)(state >> 56)` maps to the full signed 8-bit range. At runtime, each weight is dequantized on-the-fly as $w_{\text{fp32}} = w_{\text{int8}} \cdot (1/128)$ [Jacob et al., 2018]. This design captures the memory-bandwidth reduction of low-precision storage without introducing platform-specific INT8/AMX SIMD kernels.

为保证跨平台与跨构建的比特级精确复现，权重由固定种子的 64 位线性同余生成器（LCG）[Park & Miller, 1988; Knuth, 1997] 合成：

$$
x_{n+1} = a \cdot x_{n} + c \pmod{2^{64}}, \quad a = 6364136223846793005, \quad c = 1442695040888963407
$$

- **FP32 模式：** `float = (((state >> 33) / 16777216.0f) * 2.0f - 1.0f)`，映射到 $[-1, 1]$。
- **INT8 模式：** `int8_t = (int8_t)(state >> 56)`，映射至完整有符号 8 位范围。运行时每权重以 $w_{\text{fp32}} = w_{\text{int8}} \cdot (1/128)$ 进行反量化 [Jacob et al., 2018]。该设计在不引入平台相关 INT8/AMX SIMD 内核的前提下，复现了低精度存储的内存带宽缩减优势。

---

## 3. Benchmark Protocol

### 3.1 Prefill Benchmark

Each thread executes a single-token forward pass through all layers with no KV-cache (`context_len = 0`). The measured wall-clock time is then extrapolated to a 128-token sequence throughput:

$$
\text{Tokens/s} = \frac{128 \times N}{T_{\text{elapsed}} / 1000}
$$

where $N$ is the number of threads. This extrapolation assumes perfect scaling of the GEMV compute and is consistent with the kernel implementation.

每个线程在无 KV-Cache（`context_len = 0`）条件下执行单 token 的全层前向传播。测得的墙钟时间随后外推为 128-token 序列吞吐：

$$
\text{Tokens/s} = \frac{128 \times N}{T_{\text{elapsed}} / 1000}
$$

其中 $N$ 为线程数。该外推假设 GEMV 计算具有理想扩展性，与内核实现保持一致。

### 3.2 Decode Benchmark

Each thread is given a private KV-cache pre-filled to the target context length $c$. The engine executes one full layer stack with `current_step = c - 1`, forcing a complete read of all $c$ historical K/V pairs per layer. The reported metric is milliseconds per token:

$$
\text{ms/token} = \frac{T_{\text{elapsed}}}{N}
$$

Context lengths tested are **128 and 512**.

每个线程获得一个已预填充至目标上下文长度 $c$ 的私有 KV-Cache。引擎以 `current_step = c - 1` 执行完整层堆栈，迫使每层读取全部 $c$ 个历史 K/V 对。报告指标为每 token 毫秒数：

$$
\text{ms/token} = \frac{T_{\text{elapsed}}}{N}
$$

测试的上下文长度为 **128 和 512**。

### 3.3 Thread Scaling and Warm-up

Thread counts follow a power-of-two progression up to the maximum logical cores reported by the operating system, capped by an 80 % RAM safety heuristic to prevent swapping. Before measurement, one untimed warm-up pass (one Prefill plus one Decode iteration at 128 context) is executed to stabilize cache and branch-predictor state.

线程数量按 2 的幂次递增至操作系统报告的最大逻辑核心数，并受限于 80 % RAM 安全启发式以避免交换。测量前执行一次不计时的预热（一次 Prefill 加一次 128 上下文的 Decode），以稳定缓存与分支预测器状态。

---

## 4. Architecture

| Module | Responsibility |
|---|---|
| `cpu_detector` | Queries CPU topology (logical / physical cores) and SIMD capability flags (AVX2, AVX-512, AMX, NEON) via CPUID. Informational only; the kernel runs scalar C++. |
| `weight_generator` | Deterministic LCG synthesis of row-major FP32 and INT8 weight buffers. |
| `inference_engine` | Scalar transformer-layer kernel: QKV GEMV, KV-cache attention, gated FFN, residual add. Supports FP32 and runtime-dequantized INT8 paths. |
| `benchmark` | Lightweight result container (throughput / latency vectors). Orchestration logic resides in `main.cpp`. |
| `gui_chart` | Win32 GDI dashboard rendering up to four line-chart series. Skipped when `NO_GUI=1`. |

| 模块 | 职责 |
|---|---|
| `cpu_detector` | 通过 CPUID 查询 CPU 拓扑（逻辑/物理核心）与 SIMD 能力标志（AVX2、AVX-512、AMX、NEON）。仅用于信息展示；内核仍以标量 C++ 运行。 |
| `weight_generator` | 基于确定性 LCG 合成行优先的 FP32 与 INT8 权重缓冲区。 |
| `inference_engine` | 标量 Transformer 层内核：QKV GEMV、KV-Cache 注意力、门控 FFN、残差相加。支持 FP32 与运行时反量化 INT8 路径。 |
| `benchmark` | 轻量级结果容器（吞吐/延迟向量）。编排逻辑位于 `main.cpp`。 |
| `gui_chart` | Win32 GDI 仪表板，最多渲染四条折线序列。设置 `NO_GUI=1` 时跳过。 |

---

## 5. Model Configurations

| Preset | `dim` | `n_layers` | `hidden_dim` | `context_len` | FP32 Weights | INT8 Weights |
|---|---|---|---|---|---|---|
| small | 2048 | 16 | 8192 | 4096 | ~1 GB | ~256 MB |
| medium | 3072 | 24 | 12288 | 4096 | ~3 GB | ~768 MB |
| large | 4096 | 32 | 16384 | 4096 | ~8 GB | ~2 GB |

---

## 6. Build and Run

Requires CMake $\ge$ 3.15 and a C++17-compliant compiler. Zero external dependencies.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./icemark --model-size small --precision fp32
```

CLI options:

- `--model-size small|medium|large` — Select model scale (default: `small`).
- `--precision fp32|int8|both` — Select weight precision (default: `fp32`).

需要 CMake $\ge$ 3.15 与 C++17 兼容编译器。零外部依赖。

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./icemark --model-size small --precision fp32
```

命令行选项：

- `--model-size small|medium|large` — 选择模型规模（默认：`small`）。
- `--precision fp32|int8|both` — 选择权重量化精度（默认：`fp32`）。

---

## 7. Validity and Limitations

**Directional accuracy** is high: thread-scaling curves, context-length latency decay, and INT8 speed-up trends are qualitatively consistent with published observations from optimized inference engines [Xiao et al., 2023; Kwon et al., 2023].

**Absolute accuracy** is intentionally low. The kernel is implemented in scalar C++ without SIMD intrinsics, BLAS, or fused operations. It is a synthetic simulation, not a substitute for production runtimes such as llama.cpp or ONNX Runtime.

**方向准确性较高**：线程扩展曲线、上下文长度延迟衰减以及 INT8 加速趋势，与优化推理引擎的已发表观测结果在定性层面一致 [Xiao et al., 2023; Kwon et al., 2023]。

**绝对准确性刻意保持较低水平**。内核以标量 C++ 实现，未使用 SIMD 指令、BLAS 或融合算子。它是一个合成模拟工具，不能替代 llama.cpp 或 ONNX Runtime 等生产级运行时。

---

## 8. References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65–76.
- Park, S. K., & Miller, K. W. (1988). Random number generators: good ones are hard to find. *Communications of the ACM*, 31(10), 1192–1201.
- Knuth, D. E. (1997). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms* (3rd ed.). Addison-Wesley.
- Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2704–2713.
- Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). SmoothQuant: accurate and efficient post-training quantization for large language models. *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 38087–38099.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with pagedattention. *Proceedings of the 29th Symposium on Operating Systems Principles*, 611–626.
