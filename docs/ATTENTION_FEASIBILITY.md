# Feasibility: XMX low-precision / sparse attention for long-context speedup

**Question:** build a custom SYCL kernel that exploits XMX for sparse attention
(skipping zero-valued weights) using INT2/INT4, to speed up long context in
mythd/Qwen3-VL on the Arc B70.

**Goal (chosen):** *real* production long-context speedup — not a demo artifact.

**TL;DR:** Two parts of the original framing don't hold (XMX has no native
INT2/INT4; you can't skip attention zeros you haven't computed yet). More
importantly, **llama.cpp's SYCL backend already has a mature flash-attention kernel
with INT8 (q8_0) quantized KV support** — and for *decode* at long context the
bottleneck is **KV-cache bandwidth, not attention matmul**. So the highest-ROI,
near-zero-risk win is to **enable q8_0 KV in mythd today**. A custom XMX attention
kernel is high-effort, helps mainly *prefill*, and competes with a mature kernel —
better justified as research/pitch than as the production long-context fix.

---

## 1. Two hard constraints on the original idea

1. **XMX has no native INT2/INT4 compute.** On Xe2 the smallest integer operand
   the DPAS/XMX matrix engine exposes via SYCL `joint_matrix` is **INT8** (same
   wall as the BitNet work — see [[project_bitnet_arc]]). "INT2/INT4 on XMX" can
   only mean *storage* (unpacked to INT8 in registers) — bandwidth savings, not
   matrix-engine cycles. For attention, INT2 is also a quality non-starter; **INT8
   is the realistic floor** (what SageAttention uses), INT4 is research-frontier.

2. **Dynamic "skip zero attention weights" is not directly exploitable.** Weights
   are only ~zero *after* softmax, but softmax needs the full `QK^T` scores —
   which are dense to compute. You can't skip the matmul based on zeros you don't
   know yet. What *is* exploitable: **structured** sparsity (sliding-window /
   block, decided ahead of time) or **predicted** sparsity (a cheap first pass /
   heavy-hitter selection — Quest/H2O), both with quality tradeoffs.

## 2. What the engine already provides (don't reinvent)

`ggml-sycl` (this llama.cpp build, `ggml/src/ggml-sycl/`) has a **mature flash
attention** implementation: `fattn-tile.hpp` (59 KB), `fattn-vec.hpp` (28 KB),
`fattn-common.hpp` (45 KB), `fattn.cpp`. It:
- **already supports INT8 (q8_0) quantized KV** — `vec_dot_fattn_vec_KQ_q8_0`,
  `dequantize_V_q8_0`, q8_0 K/V paths;
- references `joint_matrix`/XMX in the SYCL backend;
- flash attention **auto-enables** on the B70 (seen in mythd's load log).

So the "low-precision KV storage" half of the idea is **already implemented** —
mythd just doesn't use it (it loads with default **f16** KV).

## 3. Where the long-context bottleneck actually is (this model)

Qwen3-(VL-)30B-A3B is **GQA (4 KV heads)** + **MoE (3B active)**. Split by phase:

- **Decode (per generated token):** dominated by **memory bandwidth** — reading the
  whole KV cache every step, plus streaming the active expert weights. Attention
  *matmul* is tiny here (one query row). → The lever is **KV bandwidth/size**, not
  a faster attention matmul.
- **Prefill (processing the prompt):** attention matmul (`QK^T`, `P·V`) is O(n²)
  and *does* matter for long prompts — this is the "first-token-slow on a long/
  grounded prompt" cost. → A faster attention matmul (INT8/XMX) or block-sparsity
  helps **here**.

Implication: a custom XMX attention-matmul kernel mostly speeds **prefill**.
Long-context **decode** (the steady-state chat experience) is won by **shrinking KV
bandwidth** — i.e. q8_0 KV — which already exists.

## 4. Prior art

- **SageAttention** (arXiv 2410.02367): INT8 `QK^T` (per-channel K smoothing for
  outliers) + FP16 `P·V`; ~2.1–2.7× over FlashAttention2, ~52% of INT8 peak,
  negligible end-to-end quality loss. CUDA + a ROCm port — **no SYCL/Arc port**
  (this is the gap a kernel would fill). Speeds the attention *matmul* → prefill.
- **Quest** (2406.10774) / **H2O** (NeurIPS'23): query-aware sparsity / heavy-hitter
  KV selection — reduce attention work at long context **without retraining**, with
  quality tradeoffs (error accumulates over long generation). This is the credible
  "skip the (predicted) zeros" path; it's a *selection* layer over FA, not an XMX
  kernel per se.

## 5. Options, ranked by ROI for *production long-context speedup*

| # | Option | Effort | Payoff | Risk | Phase helped |
|---|--------|--------|--------|------|--------------|
| **A** | **Enable q8_0 KV cache** (already in engine; set `type_k/type_v` in the shim) | **XS** | **High** (½ KV memory+bandwidth → ~2× context for the same VRAM, faster long-ctx decode) | Low (tiny quality hit; well-trodden) | Decode + capacity |
| B | Tune prefill batch (`n_ubatch`/`n_batch`) for the B70 | XS | Low–Med (prefill throughput) | Low | Prefill |
| C | Port **SageAttention-style INT8 `QK^T`** to SYCL/XMX (fork ggml-sycl FA) | **L** | Med (prefill of long prompts; novel-for-Arc) | Med (numerics/outliers; must beat mature FA) | Prefill |
| D | **Quest/H2O-style** query-aware KV selection in ggml-sycl FA | **L** | Med–High at *very* long ctx (decode) | Med–High (quality, error accumulation) | Decode (long) |

## 6. Recommendation

1. **Do A now** (q8_0 KV) and **measure**. It's the cheapest, lowest-risk, and
   most likely the biggest *real* long-context win for steady-state chat (decode is
   bandwidth-bound). Likely also lets you push n_ctx higher for the same VRAM.
2. **Then re-measure the bottleneck.** If long *prompt* prefill is the pain, C
   (INT8/XMX `QK^T`) is the justified kernel project — and doubles as a genuine
   Intel-pitch artifact (first SYCL/Arc SageAttention). If steady-state *decode at
   very long context* is the pain, D (query-aware sparsity) is the lever.
3. A bespoke INT2/INT4 XMX "skip-the-zeros" kernel as first framed is **not** the
   best production ROI: INT2/INT4 isn't native to XMX, dynamic zero-skipping isn't
   knowable pre-softmax, and it would have to out-engineer a mature FA for a model
   that's bandwidth-bound (not matmul-bound) at decode.

## 7. Expected gains (estimates — validate with a depth benchmark)

- **q8_0 KV (A):** KV bytes/token ≈ halved vs f16 → ~**2× context capacity** for the
  same VRAM; long-context **decode** speedup roughly tracks the KV-bandwidth
  reduction (model/depth dependent; measure with `llama-bench -fa 1 -ctk q8_0
  -ctv q8_0 -d <depth>` vs f16). Tiny perplexity impact in practice.
- **INT8 `QK^T` (C):** up to ~2× the attention-matmul throughput → meaningful only
  on **long-prompt prefill** (decode barely uses it).
- **Quest/H2O (D):** attention work at long ctx cut by the sparsity ratio (e.g.
  attend to top-k of N keys), at a tunable quality cost.

*Benchmarks here need the GPU; run when mythd can be paused, or use the 8B VL model
to avoid contending with the live 30B service.*

## References
- SageAttention — https://arxiv.org/html/2410.02367v1 · https://github.com/thu-ml/SageAttention
- Quest — https://arxiv.org/pdf/2406.10774
- H2O — NeurIPS 2023 Heavy-Hitter Oracle
- llama.cpp SYCL flash attention — `ggml/src/ggml-sycl/fattn-*.{cpp,hpp}` (q8_0 KV supported)
